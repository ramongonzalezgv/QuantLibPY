from abc import ABC, abstractmethod
from typing import Union, Dict, Optional, Any
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

    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """
        Método obligatorio para devolver los parámetros esenciales 
        del contrato (K, T, etc.) para la clave de caché.
        """
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
    def F(self) -> Optional[float]:
        """Forward price (si está definido)."""
        return self._F
    
    def get_parameters(self) -> Dict[str, Any]:
        """
        Devuelve un diccionario de parámetros esenciales para la clave de caché.
        """
        # Excluimos self.S (precio spot), self.premium y self._F ya que no son parte
        # de la definición intrínseca del contrato de opción.
        return {
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

class AsianOption(FinancialProduct):
    """
    Opción Asiática (generalmente Average Price).

    El payoff depende del promedio de los precios del activo subyacente
    observados a lo largo de la vida de la opción.
    """
    
    def __init__(self, 
                 S: float, 
                 K: float, 
                 T: Union[float, str], 
                 option_type: str = 'call', 
                 qty: int = 1,
                 averaging_type: str = 'arithmetic'):
        
        # Validaciones de entrada (similares a EuropeanOption)
        if S < 0 or K < 0:
             raise ValueError("Spot price (S) and Strike (K) must be positive")
        if option_type.lower() not in ["call", "put"]:
             raise ValueError("Option type must be 'call' or 'put'")
        if averaging_type.lower() not in ["arithmetic", "geometric"]:
             # Aunque Monte Carlo se usa mejor para Aritméticas, podemos prepararlo.
             raise ValueError("Averaging type must be 'arithmetic' or 'geometric'")

        self.S = S
        self.K = K
        self.T = T # Maturity: float (days) or str (date)
        self.option_type = option_type.lower()
        self.qty = qty
        self.averaging_type = averaging_type.lower()

    def get_parameters(self) -> Dict[str, Any]:
        """
        Devuelve un diccionario de parámetros esenciales para la clave de caché.
        """
        # Excluimos self.S.
        return {
            "K": self.K,
            "T": self.T,
            "option_type": self.option_type,
            "averaging_type": self.averaging_type,
        }

    @property
    def ttm(self) -> float:
        """Time to maturity en años. Reutiliza la lógica de EuropeanOption."""
        if isinstance(self.T, str):
            today = date.today()
            maturity_date = datetime.strptime(self.T, '%d-%m-%Y').date()
            return (maturity_date - today).days / 365.0
        else:
            # Asume T es en días si es float, y convierte a años. 
            # Esto es coherente con tu EuropeanOption.
            return self.T / 365
            
    def payoff(self, spot_prices: np.ndarray) -> np.ndarray:
        """
        Calcula el payoff de la opción asiática.
        
        Parameters
        ----------
        spot_prices : np.ndarray de forma (n_paths, n_steps + 1)
            Todos los precios simulados del activo para todas las trayectorias.
            
        Returns
        -------
        payoffs : np.ndarray de forma (n_paths,)
            El payoff para cada trayectoria.
        """
        
        # 1. Calcular el precio promedio (Average Price)
        if self.averaging_type == 'arithmetic':
            # Promedio Aritmético: (S1 + S2 + ... + Sn) / n
            Average_Price = np.mean(spot_prices, axis=1)
        elif self.averaging_type == 'geometric':
            # Promedio Geométrico: (S1 * S2 * ... * Sn)^(1/n)
            # Para evitar underflow/overflow, se usa log(producto)
            # Es importante notar que las asiáticas geométricas a menudo tienen soluciones analíticas.
            
            # El promedio geométrico se calcula como exp(mean(log(prices)))
            # Debemos asegurarnos de que los precios no sean cero o negativos.
            if np.any(spot_prices <= 0):
                 # Esto puede ocurrir en ciertos esquemas de simulación de varianza estocástica
                 # Se puede manejar con una pequeña epsilon o un esquema de simulación más robusto, 
                 # pero para MC estándar es seguro asumir precios positivos.
                 raise ValueError("Precios no positivos encontrados, imposible calcular promedio geométrico.")
                 
            Average_Price = np.exp(np.mean(np.log(spot_prices), axis=1))
        
        # 2. Calcular el payoff final (similar al Europeo, pero usando el precio promedio)
        if self.option_type == 'call':
            # Payoff: max(Average_Price - K, 0)
            return self.qty * np.maximum(Average_Price - self.K, 0)
        else: # 'put'
            # Payoff: max(K - Average_Price, 0)
            return self.qty * np.maximum(self.K - Average_Price, 0)

    def info(self) -> str:
        """Información del producto."""
        # Se puede añadir una función de información similar a la de EuropeanOption
        info = f"""
            Asian {self.option_type.capitalize()} Option ({self.averaging_type.capitalize()} Average)
            --------------------------------
            Spot Price (S): {self.S}
            Strike Price (K): {self.K}
            Maturity: {self.T}
            Time to Maturity: {self.ttm * 365:.2f} days
            Quantity: {self.qty}
        """
        return info