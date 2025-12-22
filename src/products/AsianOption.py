from typing import Union, Dict, Any, Optional
from datetime import date, datetime
import numpy as np

from .FinancialProduct import FinancialProduct

from ..utils.marketcalendars.MarketCalendar import MarketCalendar

class AsianOption(FinancialProduct):
    """
    Asian Option (generally Average Price).

    The payoff depends on the average of the prices of the underlying asset
    observed over the life of the option.
    """
    
    def __init__(self, 
                 S: float, 
                 K: float, 
                 expiry_date: Union[str, date],
                 valuation_date: Union[str, date, None] = None,
                 calendar: Optional[MarketCalendar] = None,
                 option_type: str = 'call', 
                 qty: int = 1,
                 averaging_type: str = 'arithmetic'):
        
        # Input validations (similar to EuropeanOption)
        if S < 0 or K < 0:
             raise ValueError("Spot price (S) and Strike (K) must be positive")
        if option_type.lower() not in ["call", "put"]:
             raise ValueError("Option type must be 'call' or 'put'")
        if averaging_type.lower() not in ["arithmetic", "geometric"]:
             # Although Monte Carlo is better for Arithmetic, we can prepare it.
             raise ValueError("Averaging type must be 'arithmetic' or 'geometric'")

        self.S = S
        self.K = K
        self.calendar = calendar
        self.option_type = option_type.lower()
        self.qty = qty
        self.averaging_type = averaging_type.lower()

         # --- Date Handling ---
        self.expiry_date = self._parse_date(expiry_date)
        
        if valuation_date is None:
            self.valuation_date = date.today()
        else:
            self.valuation_date = self._parse_date(valuation_date)

        if self.valuation_date > self.expiry_date:
            raise ValueError(f"Valuation date ({self.valuation_date}) cannot be after expiry ({self.expiry_date})")

    def _parse_date(self, date_input: Union[str, date]) -> date:
        """Helper to ensure we always work with date objects."""
        if isinstance(date_input, str):
            # Assumes ISO format YYYY-MM-DD or DD-MM-YYYY, adjust as needed
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
        """Returns the raw number of days based on the chosen calendar."""
        if self.calendar:
            return self.calendar.count_days(self.valuation_date, self.expiry_date)
        else:
            # Default: Actual calendar days
            return (self.expiry_date - self.valuation_date).days

    @property
    def ttm(self) -> float:
        """
        Time to maturity as a year fraction.
        """
        days = self.days_to_maturity
        
        if self.calendar:
            # Example: 10 business days / 252
            return max(0.0, days / self.calendar.days_in_year)
        else:
            # Example: 14 calendar days / 365
            return max(0.0, days / 365.0)

    def get_parameters(self) -> Dict[str, Any]:
        """
        Returns a dictionary of essential parameters for the cache key.
        """
        # We exclude self.S.
        return {
            "S": self.S,
            "K": self.K,
            "expiry_date": self.expiry_date,
            "option_type": self.option_type,
            "averaging_type": self.averaging_type,
        }
            
    def payoff(self, spot_prices: np.ndarray) -> np.ndarray:
        """
        Calculates the payoff of the Asian option.
        
        Parameters
        ----------
        spot_prices : np.ndarray of shape (n_paths, n_steps + 1)
            All simulated prices of the asset for all paths.
            
        Returns
        -------
        payoffs : np.ndarray of shape (n_paths,)
            The payoff for each path.
        """
        
        # 1. Calculate the average price (Average Price)
        if self.averaging_type == 'arithmetic':
            # Arithmetic Average: (S1 + S2 + ... + Sn) / n
            Average_Price = np.mean(spot_prices, axis=1)
        elif self.averaging_type == 'geometric':
            # Geometric Average: (S1 * S2 * ... * Sn)^(1/n)
            # To avoid underflow/overflow, we use log(product)
            # It is important to note that geometric Asian options often have analytical solutions.
            
            # The geometric average is calculated as exp(mean(log(prices)))
            # We need to ensure that the prices are not zero or negative.
            if np.any(spot_prices <= 0):
                 # This can happen in certain stochastic variance simulation schemes
                 # It can be handled with a small epsilon or a more robust simulation scheme, 
                 # but for standard MC it is safe to assume positive prices.
                 raise ValueError("Positive prices not found, impossible to calculate geometric average.")
                 
            Average_Price = np.exp(np.mean(np.log(spot_prices), axis=1))
        
        # 2. Calculate the final payoff (similar to European, but using the average price)
        if self.option_type == 'call':
            # Payoff: max(Average_Price - K, 0)
            return self.qty * np.maximum(Average_Price - self.K, 0)
        else: # 'put'
            # Payoff: max(K - Average_Price, 0)
            return self.qty * np.maximum(self.K - Average_Price, 0)

    def info(self) -> str:
        """Information about the product."""
        info = f"""
        Asian {self.option_type.capitalize()} Option ({self.averaging_type.capitalize()} Average)
        --------------------------------
        Spot Price (S): {self.S}
        Strike Price (K): {self.K}
        Maturity: {self.expiry_date}
        Time to Maturity: {self.ttm * 365:.2f} days
        Quantity: {self.qty}
        """
        return info