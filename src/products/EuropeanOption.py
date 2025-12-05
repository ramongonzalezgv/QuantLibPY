from typing import Union, Dict, Optional, Any
from datetime import date, datetime, timedelta
import numpy as np

from .FinancialProduct import FinancialProduct

class EuropeanOption(FinancialProduct):
    """Vanilla European Option."""
    
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
        """Time to maturity in years."""

        if isinstance(self.T, str):
            today = date.today()
            maturity_date = datetime.strptime(self.T, '%d-%m-%Y').date()
            return (maturity_date - today).days / 365.0
        else:
            return self.T / 365
    
    @property
    def maturity_date(self) -> Union[str, date]:
        """Expiry date."""

        if isinstance(self.T, str):
            return self.T
        else:
            return date.today() + timedelta(days=self.T)
    
    @property
    def F(self) -> Optional[float]:
        """Forward price (if defined)."""

        return self._F
    
    def get_parameters(self) -> Dict[str, Any]:
        """
        Devuelve un diccionario de parámetros esenciales para la clave de caché.
        """
        # Excluimos self.S (precio spot), self.premium y self._F ya que no son parte
        # de la definición intrínseca del contrato de opción.
        return {
            "S": self.S,
            "K": self.K,
            # Usamos T (el input original) en lugar de ttm, para mantener la clave estable.
            # Si T es una fecha (str), se convierte a str para JSON.
            "T": self.T,
            "option_type": self.option_type,
            "qty": self.qty,
        }
    
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