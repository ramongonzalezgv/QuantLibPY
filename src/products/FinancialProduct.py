from abc import ABC, abstractmethod
from typing import Dict, Any
import numpy as np

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