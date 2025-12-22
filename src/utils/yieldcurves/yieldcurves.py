import numpy as np
from datetime import date, timedelta
from typing import List, Tuple, Dict, Optional, Union
from scipy.interpolate import interp1d
from abc import ABC, abstractmethod


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
        Calcula el factor de descuento hasta una fecha objetivo.
        
        Args:
            target_date: Fecha objetivo o tiempo en años
        
        Returns:
            Factor de descuento P(0, T)
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
        Calcula la tasa cero (continuously compounded) hasta una fecha.
        
        Returns:
            Tasa cero r tal que DF(T) = exp(-r * T)
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


class FlatYieldCurve(YieldCurve):
    """Curva plana (flat) - todos los plazos tienen la misma tasa."""
    
    def __init__(self, reference_date: date, rate: float, day_count_convention: str = 'ACT/365'):
        """
        Args:
            reference_date: Fecha de referencia
            rate: Tasa de interés constante (continuously compounded)
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
            reference_date: Fecha de referencia
            pillars: Lista de (fecha, tasa_cero) para los pilares
            interpolation: Método de interpolación ('linear', 'cubic')
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
            raise ValueError("Se necesitan al menos 2 pilares")
        
        kind = 'linear' if interpolation == 'linear' else 'cubic'
        self.interpolator = interp1d(
            self.pillar_times, 
            self.pillar_rates,
            kind=kind,
            fill_value='extrapolate',
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


class BootstrappedYieldCurve(YieldCurve):
    """
    Curve constructed using boostrappibg from market instruments.
    """
    
    def __init__(self,
                 reference_date: date,
                 instruments: List[Dict],
                 day_count_convention: str = 'ACT/365'):
        """
        Args:
            reference_date: Date of valuation
            instruments: List of market instruments
                Example: [
                    {'type': 'deposit', 'maturity': date(...), 'rate': 0.02},
                    {'type': 'swap', 'maturity': date(...), 'rate': 0.025, 'frequency': 2}
                ]
        """
        super().__init__(reference_date, day_count_convention)
        
        # Bootstrap
        self.pillar_times = []
        self.pillar_dfs = []
        
        self._bootstrap(instruments)
        
        # Crear interpolador de discount factors
        if len(self.pillar_times) >= 2:
            # Interpolar log(DF) para mejor comportamiento
            log_dfs = np.log(self.pillar_dfs)
            self.interpolator = interp1d(
                self.pillar_times,
                log_dfs,
                kind='linear',
                fill_value='extrapolate',
                bounds_error=False
            )
        else:
            raise ValueError("At least 2 intruments are needed")
    
    def _bootstrap(self, instruments: List[Dict]):
        """Implements the curve boostrapping"""
        # Order by maturity
        instruments = sorted(instruments, key=lambda x: x['maturity'])
        
        for inst in instruments:
            maturity_time = self.year_fraction(self.reference_date, inst['maturity'])
            
            if inst['type'] == 'deposit':
                # Deposit: DF = 1 / (1 + r * T)
                rate = inst['rate']
                df = 1.0 / (1.0 + rate * maturity_time)
                
            elif inst['type'] == 'swap':
                # Swap: resolve for DF that makes NPV = 0
                rate = inst['rate']
                frequency = inst.get('frequency', 2)
                df = self._bootstrap_swap(rate, maturity_time, frequency)
                
            else:
                raise ValueError(f"Unknown market instrument: {inst['type']}")
            
            self.pillar_times.append(maturity_time)
            self.pillar_dfs.append(df)
    
    def _bootstrap_swap(self, swap_rate: float, maturity_time: float, frequency: int) -> float:
        """
        Resolve for the discount factor of the swap.
        
        NPV_fixed = sum(rate * delta * DF_i) + DF_n
        NPV_float = 1.0
        
        For par swap: NPV_fixed = NPV_float
        => DF_n = 1 - swap_rate * sum(delta * DF_i)
        """
        # Generate payment times
        delta = 1.0 / frequency
        payment_times = np.arange(delta, maturity_time + delta/2, delta)
        
        # Calculate sum of delta * DF for previous payments
        annuity = 0.0
        for t in payment_times[:-1]:
            df = self.discount_factor(t)
            annuity += delta * df
        
        # Resolve for final DF
        df_final = (1.0 - swap_rate * annuity) / (1.0 + swap_rate * delta)
        
        return df_final
    
    def discount_factor(self, target_date: Union[date, float]) -> float:
        """Interpola discount factor."""
        if isinstance(target_date, (int, float)):
            T = target_date
        else:
            T = self.year_fraction(self.reference_date, target_date)
        
        if T <= 0:
            return 1.0
        
        # Interpolar log(DF)
        log_df = float(self.interpolator(T))
        return np.exp(log_df)


