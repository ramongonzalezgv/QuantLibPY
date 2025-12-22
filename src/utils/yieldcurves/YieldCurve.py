import numpy as np
from datetime import date, timedelta
from typing import List, Tuple, Dict, Optional, Union
from scipy.interpolate import interp1d
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt


class YieldCurve(ABC):
    """
    Clase base para curvas de tipos de interés.
    Permite calcular discount factors, forward rates, y swap rates.
    """
    
    def __init__(self, reference_date: date, day_count_convention: str = 'ACT/365'):
        """
        Args:
            reference_date: Fecha de referencia de la curva
            day_count_convention: Convención de conteo de días
        """
        self.reference_date = reference_date
        self.day_count_convention = day_count_convention
    
    @abstractmethod
    def discount_factor(self, target_date: Union[date, float]) -> float:
        """
        Computes the discount factor until the given date.
        
        Args:
            target_date: Target date in years
        
        Returns:
            Discount factor P(0, T)
        """
        pass
    
    def year_fraction(self, start_date: date, end_date: date) -> float:
        """Calcula la fracción de año entre dos fechas."""
        days = (end_date - start_date).days
        
        if self.day_count_convention == 'ACT/365':
            return days / 365.0
        elif self.day_count_convention == 'ACT/360':
            return days / 360.0
        elif self.day_count_convention == '30/360':
            # Simplified 30/360 calculation
            d1, m1, y1 = start_date.day, start_date.month, start_date.year
            d2, m2, y2 = end_date.day, end_date.month, end_date.year
            return ((y2 - y1) * 360 + (m2 - m1) * 30 + (d2 - d1)) / 360.0
        else:
            return days / 365.0
    
    def zero_rate(self, target_date: Union[date, float]) -> float:
        """
        Calculates the continuously compounded zero rate until a given date.
        
        Returns:
            Zero rate r such that DF(T) = exp(-r * T)
        """
        if isinstance(target_date, (int, float)):
            T = target_date
        else:
            T = self.year_fraction(self.reference_date, target_date)
        
        df = self.discount_factor(T)
        
        if T <= 0 or df <= 0:
            return 0.0
        
        return -np.log(df) / T
    
    def forward_rate(self, start_date: date, end_date: date) -> float:
        """
        Calcula la tasa forward simplemente compuesta entre dos fechas.
        
        Formula: F(t1, t2) = (DF(t1) / DF(t2) - 1) / delta
        
        Returns:
            Forward rate
        """
        T1 = self.year_fraction(self.reference_date, start_date)
        T2 = self.year_fraction(self.reference_date, end_date)
        
        df1 = self.discount_factor(T1)
        df2 = self.discount_factor(T2)
        delta = T2 - T1
        
        if delta <= 0:
            return 0.0
        
        return (df1 / df2 - 1.0) / delta
    
    def swap_rate(self, 
                  start_date: date, 
                  end_date: date, 
                  payment_frequency: int = 2) -> float:
        """
        Calcula la tasa swap (par rate) entre dos fechas.
        
        La tasa swap es la tasa fija que hace que el NPV del swap sea cero.
        
        Formula: S = (DF(start) - DF(end)) / Annuity
        
        Args:
            start_date: Fecha de inicio del swap
            end_date: Fecha final del swap
            payment_frequency: Frecuencia de pagos por año (2 = semi-annual)
        
        Returns:
            Swap rate (tasa fija)
        """
        # Generar fechas de pago
        payment_dates = self._generate_payment_dates(
            start_date, end_date, payment_frequency
        )
        
        # Calcular annuity (suma de DF × delta)
        annuity = 0.0
        for i in range(len(payment_dates) - 1):
            delta = self.year_fraction(payment_dates[i], payment_dates[i+1])
            T = self.year_fraction(self.reference_date, payment_dates[i+1])
            df = self.discount_factor(T)
            annuity += delta * df
        
        # Discount factors inicial y final
        T_start = self.year_fraction(self.reference_date, start_date)
        T_end = self.year_fraction(self.reference_date, end_date)
        df_start = self.discount_factor(T_start)
        df_end = self.discount_factor(T_end)
        
        if annuity <= 0:
            return 0.0
        
        return (df_start - df_end) / annuity
    
    def _generate_payment_dates(self, 
                               start_date: date, 
                               end_date: date, 
                               frequency: int) -> List[date]:
        """Genera las fechas de pago para un swap."""
        dates = [start_date]
        current = start_date
        
        # Periodo en meses
        period_months = 12 // frequency
        
        while current < end_date:
            # Añadir periodo
            next_date = self._add_months(current, period_months)
            if next_date > end_date:
                next_date = end_date
            dates.append(next_date)
            current = next_date
        
        return dates
    
    def _add_months(self, start_date: date, months: int) -> date:
        """Añade meses a una fecha."""
        month = start_date.month - 1 + months
        year = start_date.year + month // 12
        month = month % 12 + 1
        day = min(start_date.day, [31, 29 if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0) else 28, 
                                    31, 30, 31, 30, 31, 31, 30, 31, 30, 31][month-1])
        return date(year, month, day)


