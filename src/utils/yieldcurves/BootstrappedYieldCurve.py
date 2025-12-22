import numpy as np
import re
from datetime import date
from typing import List, Dict, Union, cast, Any
from scipy.interpolate import interp1d, PchipInterpolator
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from .YieldCurve import YieldCurve

   
class BootstrappedYieldCurve(YieldCurve):
    def __init__(self, 
                 reference_date: date, 
                 instruments: List[Dict], 
                 interpolation: str = 'monotone',
                 day_count_convention: str = 'ACT/365'):
        # Initialize the parent class
        super().__init__(reference_date, day_count_convention)
        
        # Internal state for bootstrapping
        self.pillar_times = [0.0]
        self.pillar_dfs = [1.0]
        self.interpolator = None
        self.interpolation = interpolation.lower()
        
        # Run the bootstrap
        self._bootstrap(instruments)
    
    def discount_factor(self, target_date: Union[date, float]) -> float:
        if isinstance(target_date, (int, float)):
            T = target_date
        else:
            T = self.year_fraction(self.reference_date, target_date)

        if T <= 0: 
            return 1.0
        if self.interpolator is None:
            return 1.0

        # inside known range: use interpolator
        if T <= self._t_max:
            ln_df = float(self.interpolator(T))
        else:
            # flat instantaneous forward extrapolation beyond last liquid point)
            ln_df = self._ln_df_last - self._last_forward * (T - self._t_max)

        return float(np.exp(ln_df))

    def _bootstrap(self, instruments: List[Dict]):
        # Sort by maturity using the parent's year_fraction
        instruments = sorted(instruments, key=lambda x: x['maturity'])
        
        # Start with the T=0 point
        self._update_interpolator()
        
        for inst in instruments:
            T_mat = self.year_fraction(self.reference_date, inst['maturity'])
            
            if inst['type'] == 'deposit':
                df = 1.0 / (1.0 + inst['rate'] * T_mat)
            elif inst['type'] == 'swap':
                df = self._bootstrap_swap(inst['rate'], inst['maturity'], inst.get('frequency', 2))
            
            self.pillar_times.append(T_mat)
            self.pillar_dfs.append(df)
            self._update_interpolator()

    def _bootstrap_swap(self, market_rate: float, maturity_date: date, frequency: int) -> float:
        """Solves for the final DF that makes the Swap Par Rate equal market_rate."""
        # Use parent's date generation
        payment_dates = self._generate_payment_dates(self.reference_date, maturity_date, frequency)
        
        annuity = 0.0
        # Sum up all intermediate payments using CURRENT interpolator
        for i in range(len(payment_dates) - 1):
            d1, d2 = payment_dates[i], payment_dates[i+1]
            delta = self.year_fraction(d1, d2)
            T = self.year_fraction(self.reference_date, d2)
            
            # The last payment is the one we are solving for, 
            # so we only sum up to the penultimate payment
            if d2 < maturity_date:
                annuity += delta * self.discount_factor(T)
            else:
                last_delta = delta

        # Solve for DF_n: MarketRate = (1 - DF_n) / (Annuity_prev + last_delta * DF_n)
        # DF_n = (1 - MarketRate * Annuity_prev) / (1 + MarketRate * last_delta)
        return (1.0 - market_rate * annuity) / (1.0 + market_rate * last_delta)

    def _update_interpolator(self):
        n_points = len(self.pillar_times)
        if n_points < 2:
            self.interpolator = lambda t: 0.0
            # fallback values
            self._t_max = 0.0
            self._ln_df_last = 0.0
            self._last_forward = 0.0
            return

        # ensure arrays sorted
        times = np.array(self.pillar_times)
        log_dfs = np.log(np.array(self.pillar_dfs))

        if self.interpolation == 'monotone':
            self.interpolator = PchipInterpolator(times, log_dfs, extrapolate=False)
        else:
            kind = self.interpolation
            if kind == 'cubic' and n_points < 4:
                kind = 'linear'
            elif kind == 'quadratic' and n_points < 3:
                kind = 'linear'
            self.interpolator = interp1d(times, log_dfs, kind=kind,
                                         fill_value=cast(Any,'extrapolate'), bounds_error=False)

        # store last-point info for safe extrapolation (constant forward)
        self._t_max = float(times[-1])
        self._ln_df_last = float(log_dfs[-1])
        # approximate derivative on last interval: d lnDF / dt
        dt = times[-1] - times[-2]
        if dt <= 0:
            slope = 0.0
        else:
            slope = (log_dfs[-1] - log_dfs[-2]) / dt
        self._last_forward = -float(slope)  # f = -d ln DF / dt

    def plot(self, curves_to_plot: List[str] = ['zero', '6M']):
        """
        Flexible plotting for Zero and Forward curves.
        Example: curves_to_plot=['zero', '3M', '6M', '1Y']
        """
        if not self.pillar_times:
            return

        # 1. Parse the requested tenors to find the maximum offset
        # This prevents plotting past our data horizon
        max_t = max(self.pillar_times)
        max_offset_months = 0
        parsed_requests = []

        for req in curves_to_plot:
            req_clean = req.lower().strip()
            if req_clean == 'zero':
                parsed_requests.append(('zero', 0, 'Zero Rate'))
            else:
                # Regex to find number and period (M or Y)
                match = re.match(r'(\d+)([my])', req_clean)
                if match:
                    val, unit = int(match.group(1)), match.group(2)
                    months = val if unit == 'm' else val * 12
                    max_offset_months = max(max_offset_months, months)
                    parsed_requests.append(('forward', months, f'{req.upper()} Forward'))
                else:
                    print(f"Warning: Could not parse curve request '{req}'")

        # 2. Adjust time range to avoid extrapolating forwards past the last pillar
        t_smooth = np.linspace(0.01, max_t - (max_offset_months / 12.0), 200)
        
        plt.figure(figsize=(12, 7))
        
        # 3. Calculate and plot each curve
        for type, months, label in parsed_requests:
            y_values = []
            for t in t_smooth:
                if type == 'zero':
                    y_values.append(self.zero_rate(t))
                else:
                    # Calculate start/end dates using parent's utility
                    start_date = self._add_months(self.reference_date, int(t * 12))
                    end_date = self._add_months(start_date, months)
                    y_values.append(self.forward_rate(start_date, end_date))
            
            plt.plot(t_smooth, y_values, label=label, linewidth=2)

        # 4. Plot original market pillars
        t_pillars = [t for t in self.pillar_times if t > 0]
        z_pillars = [self.zero_rate(t) for t in t_pillars]
        plt.scatter(t_pillars, z_pillars, color='black', marker='x', 
                    label='Market Pillars', zorder=10)

        # Formatting
        plt.title(f"Yield Curve Analysis: {', '.join(curves_to_plot)}", fontsize=14)
        plt.xlabel("Years (T)")
        plt.ylabel("Rate (%)")
        plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.2%}'))
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend()
        plt.show()