from abc import ABC, abstractmethod
from typing import Dict, Any
import numpy as np

class FinancialProduct(ABC):
    """Base class for financial products."""
    
    @abstractmethod
    def payoff(self, spot_prices: np.ndarray) -> np.ndarray:
        """Calculates the payoff of the product."""
        pass

    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """
        Mandatory method to return essential parameters 
        of the contract (K, T, etc.) for the cache key.
        """
        pass