import numpy as np
from datetime import date
from typing import  Union
import matplotlib.pyplot as plt

from .YieldCurve import YieldCurve

class FlatYieldCurve(YieldCurve):
    """Flat yield curve - all maturities have the same rate."""
    
    def __init__(self, reference_date: date, rate: float, day_count_convention: str = 'ACT/365'):
        """
        Args:
            reference_date: Fecha de referencia
            rate: Constant interest rate (continuously compounded)
        """
        super().__init__(reference_date, day_count_convention)
        self.rate = rate
    
    def discount_factor(self, target_date: Union[date, float]) -> float:
        """DF(T) = exp(-r * T)"""
        if isinstance(target_date, (int, float)):
            T = target_date
        else:
            T = self.year_fraction(self.reference_date, target_date)
        
        return np.exp(-self.rate * T)