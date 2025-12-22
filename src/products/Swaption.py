from typing import Union, Dict, Optional, Any
from datetime import date, datetime
import numpy as np

from .FinancialProduct import FinancialProduct
from ..utils.marketcalendars.MarketCalendar import MarketCalendar


class Swaption(FinancialProduct):
    """European Swaption (option on interest rate swap)."""
    
    def __init__(self,
                 notional: float,
                 strike_rate: float,  # fixed rate K
                 tenor: float,  # swap tenor in years
                 expiry_date: Union[str, date],
                 swap_start_date: Union[str, date],
                 valuation_date: Union[str, date, None] = None,
                 calendar: Optional[MarketCalendar] = None,
                 option_type: str = 'payer',  # 'payer' or 'receiver'
                 payment_frequency: int = 2,  # semi-annual = 2
                 qty: int = 1):
        
        if notional <= 0:
            raise ValueError("Notional must be positive")
        if strike_rate < 0:
            raise ValueError("Strike rate must be non-negative")
        if tenor <= 0:
            raise ValueError("Tenor must be positive")
        if option_type.lower() not in ["payer", "receiver"]:
            raise ValueError("Option type must be 'payer' or 'receiver'")
        
        self.notional = notional
        self.K = strike_rate  # strike rate (fixed rate)
        self.tenor = tenor
        self.option_type = option_type.lower()
        self.payment_frequency = payment_frequency
        self.qty = qty
        self.calendar = calendar
        
        # Date handling
        self.expiry_date = self._parse_date(expiry_date)
        self.swap_start_date = self._parse_date(swap_start_date)
        
        if valuation_date is None:
            self.valuation_date = date.today()
        else:
            self.valuation_date = self._parse_date(valuation_date)
        
        if self.valuation_date > self.expiry_date:
            raise ValueError("Valuation date cannot be after expiry")
        if self.expiry_date > self.swap_start_date:
            raise ValueError("Expiry date cannot be after swap start date")
        
        # Forward swap rate (will be set externally or computed)
        self._F = None
    
    def _parse_date(self, date_input: Union[str, date]) -> date:
        """Helper to parse dates."""
        if isinstance(date_input, str):
            try:
                return datetime.strptime(date_input, '%Y-%m-%d').date()
            except ValueError:
                return datetime.strptime(date_input, '%d-%m-%Y').date()
        elif isinstance(date_input, datetime):
            return date_input.date()
        elif isinstance(date_input, date):
            return date_input
        else:
            raise TypeError("Date must be a string or datetime.date object")
    
    @property
    def days_to_maturity(self) -> int:
        """Days until option expiry."""
        if self.calendar:
            return self.calendar.count_days(self.valuation_date, self.expiry_date)
        else:
            return (self.expiry_date - self.valuation_date).days
    
    @property
    def ttm(self) -> float:
        """Time to option expiry as year fraction."""
        days = self.days_to_maturity
        
        if self.calendar:
            return max(0.0, days / self.calendar.days_in_year)
        else:
            return max(0.0, days / 365.0)
    
    @property
    def F(self) -> Optional[float]:
        """Forward swap rate."""
        return self._F
    
    @F.setter
    def F(self, value: float):
        """Set forward swap rate."""
        self._F = value
    
    def annuity(self, r: float) -> float:
        """
        Calculate the present value of basis point (PVBP) or annuity factor.
        This is the sum of discount factors for all payment dates.
        """
        # Calculate payment schedule
        denom = self.calendar.days_in_year if self.calendar else 365.0
        start_offset = (self.swap_start_date - self.valuation_date).days / denom
        
        # Payment period in years
        delta = 1.0 / self.payment_frequency
        
        # Number of payments
        n_payments = int(round(self.tenor * self.payment_frequency))
        
        if n_payments == 0:
            return 0.0
        
        # Payment times relative to valuation date
        payment_times = start_offset + delta * np.arange(1, n_payments + 1)
        
        # Discount factors
        discount_factors = np.exp(-r * payment_times)
        
        # Annuity = sum of (payment_period * discount_factor)
        annuity = float(np.sum(delta * discount_factors))
        
        return annuity
    
    def get_parameters(self) -> Dict[str, Any]:
        """Return parameters for cache key."""
        return {
            "notional": self.notional,
            "K": self.K,
            "tenor": self.tenor,
            "expiry_date": self.expiry_date,
            "swap_start_date": self.swap_start_date,
            "option_type": self.option_type,
            "payment_frequency": self.payment_frequency,
            "qty": self.qty,
        }
    
    def payoff(self, forward_rates: np.ndarray) -> np.ndarray:
        """
        Calculate swaption payoff at expiry.
        forward_rates: array of forward swap rates at expiry.
        """
        # Payer swaption: option to pay fixed, receive floating
        # Payoff when forward rate > strike (you pay fixed at K, receive higher floating)
        if self.option_type == 'payer':
            return self.qty * self.notional * np.maximum(forward_rates - self.K, 0)
        else:  # receiver
            return self.qty * self.notional * np.maximum(self.K - forward_rates, 0)
    
    def info(self) -> str:
        """Information about the swaption."""
        info = f"""
        {self.option_type.capitalize()} Swaption
        --------------------------------
        Notional: {self.notional:,.0f}
        Strike Rate: {self.K:.4%}
        Forward Swap Rate: {self.F:.4%} if self.F else 'Not set'
        Tenor: {self.tenor} years
        Expiry Date: {self.expiry_date}
        Swap Start: {self.swap_start_date}
        Time to Expiry: {self.ttm * 365:.2f} days
        Payment Frequency: {self.payment_frequency}x per year
        Quantity: {self.qty}
        """
        return info