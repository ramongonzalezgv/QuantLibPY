from abc import ABC, abstractmethod
from typing import Union, Dict, Optional
from datetime import date, datetime, timedelta
import numpy as np
import scipy.stats as si
from scipy.optimize import brentq
from scipy.fft import ifft
from scipy.interpolate import interp1d

# ============================================================================
# PRODUCTS: Definen los instrumentos financieros
# ============================================================================

class FinancialProduct(ABC):
    """Clase base para productos financieros."""
    
    @abstractmethod
    def payoff(self, spot_prices: np.ndarray) -> np.ndarray:
        """Calcula el payoff del producto."""
        pass


class EuropeanOption(FinancialProduct):
    """Opción Europea vanilla."""
    
    def __init__(self, 
                 S: float,
                 K: float,
                 T: Union[float, str],
                 option_type: str = 'call',
                 qty: int = 1,
                 premium: Optional[float] = None,
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
        self.T = T
        self.option_type = option_type.lower()
        self.qty = qty
        self.premium = premium
        self._F = F
    
    @property
    def ttm(self) -> float:
        """Time to maturity en años."""
        if isinstance(self.T, str):
            today = date.today()
            maturity_date = datetime.strptime(self.T, '%d-%m-%Y').date()
            return (maturity_date - today).days / 365.0
        else:
            return self.T / 365
    
    @property
    def maturity_date(self) -> Union[str, date]:
        """Fecha de vencimiento."""
        if isinstance(self.T, str):
            return self.T
        else:
            return date.today() + timedelta(days=self.T)
    
    @property
    def F(self) -> float:
        """Forward price (si está definido)."""
        return self._F
    
    def payoff(self, spot_prices: np.ndarray) -> np.ndarray:
        """Calcula el payoff a vencimiento."""
        if self.option_type == 'call':
            return self.qty * np.maximum(spot_prices - self.K, 0)
        else:
            return self.qty * np.maximum(self.K - spot_prices, 0)
    
    def info(self) -> str:
        """Información del producto."""
        info = f"""
European {self.option_type.capitalize()} Option
--------------------------------
Spot Price (S): {self.S}
Strike Price (K): {self.K}
Maturity: {self.T}
Time to Maturity: {self.ttm * 365:.2f} days
Quantity: {self.qty}
"""
        if self.premium:
            info += f"Premium: {self.premium:.4f}\n"
        return info


class AsianOption(FinancialProduct):
    """Opción Asiática (placeholder para expansión futura)."""
    
    def __init__(self, S: float, K: float, T: float, 
                 option_type: str = 'call', averaging_type: str = 'arithmetic'):
        self.S = S
        self.K = K
        self.T = T
        self.option_type = option_type
        self.averaging_type = averaging_type
    
    def payoff(self, spot_prices: np.ndarray) -> np.ndarray:
        raise NotImplementedError("AsianOption pendiente de implementación")