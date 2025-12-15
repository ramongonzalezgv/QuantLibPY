from typing import Union, Dict, Optional, Any
from datetime import date, datetime, timedelta
import numpy as np

from .FinancialProduct import FinancialProduct

from ..utils.MarketCalendar import MarketCalendar

class EuropeanOption(FinancialProduct):
    """Vanilla European Option."""
    
    def __init__(self, 
                 S: float,
                 K: float,
                 expiry_date: Union[str, date],
                 valuation_date: Union[str, date, None] = None,
                 calendar: Optional[MarketCalendar] = None,
                 option_type: str = 'call',
                 qty: int = 1,
                 F: Optional[float] = None):
        
        if S < 0:
            raise ValueError("Spot price must be positive")
        if K < 0:
            raise ValueError("Strike price must be positive")
        if not isinstance(qty, int):
            raise ValueError("Quantity must be an integer")
        if option_type.lower() not in ["call", "put"]:
            raise ValueError("Option type must be 'call' or 'put'")
        
        self.S = S
        self.K = K
        self.option_type = option_type.lower()
        self.qty = qty
        self._F = F
        self.calendar = calendar

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
    
    @property
    def F(self) -> Optional[float]:
        """Forward price (if defined)."""

        return self._F
    
    def get_parameters(self) -> Dict[str, Any]:
        """
        Returns a dictionary of essential parameters for the cache key.
        """
        # We exclude self.S (spot price), self.premium and self._F 
        # since they are not part of the intrinsic contract definition.
        return {
            "S": self.S,
            "K": self.K,
            # We use expiry_date instead of ttm to keep the cache key stable.
            "expiry_date": self.expiry_date,
            "option_type": self.option_type,
            "qty": self.qty,
        }
    
    def payoff(self, spot_prices: np.ndarray) -> np.ndarray:
        """Calculates the payoff at maturity."""
        if self.option_type == 'call':
            return self.qty * np.maximum(spot_prices - self.K, 0)
        else:
            return self.qty * np.maximum(self.K - spot_prices, 0)
    
    def info(self) -> str:
        """Information about the product."""
        info = f"""
                European {self.option_type.capitalize()} Option
                --------------------------------
                Spot Price (S): {self.S}
                Strike Price (K): {self.K}
                Maturity: {self.expiry_date}
                Time to Maturity: {self.ttm * 365:.2f} days
                Quantity: {self.qty}
        """
        return info