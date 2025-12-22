import numpy as np
from datetime import date
from typing import List, Tuple, Union, cast, Any
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

from .YieldCurve import YieldCurve

class InterpolatedYieldCurve(YieldCurve):
    """
    Curva interpolada desde un conjunto de puntos (pilares).
    Usa interpolación de tasas cero.
    """
    
    def __init__(self, 
                 reference_date: date,
                 pillars: List[Tuple[date, float]],
                 interpolation: str = 'linear',
                 day_count_convention: str = 'ACT/365'):
        """
        Args:
            reference_date: Reference date for the calculations
            pillars: List of (date, zero_rate) for the pillars
            interpolation: Interpolation method ('linear', 'cubic')
        """
        super().__init__(reference_date, day_count_convention)
        
        # Convertir fechas a year fractions
        self.pillar_times = []
        self.pillar_rates = []
        
        for pillar_date, rate in sorted(pillars):
            T = self.year_fraction(reference_date, pillar_date)
            self.pillar_times.append(T)
            self.pillar_rates.append(rate)
        
        # Crear interpolador
        if len(self.pillar_times) < 2:
            raise ValueError("At least 2 pillars are needed")
        
        kind = 'linear' if interpolation == 'linear' else 'cubic'
        self.interpolator = interp1d(
            self.pillar_times, 
            self.pillar_rates,
            kind=kind,
            fill_value=cast(Any,'extrapolate'),
            bounds_error=False
        )
    
    def discount_factor(self, target_date: Union[date, float]) -> float:
        """DF(T) = exp(-r(T) * T)"""
        if isinstance(target_date, (int, float)):
            T = target_date
        else:
            T = self.year_fraction(self.reference_date, target_date)
        
        if T <= 0:
            return 1.0
        
        # Interpolar tasa cero
        rate = float(self.interpolator(T))
        
        return np.exp(-rate * T)