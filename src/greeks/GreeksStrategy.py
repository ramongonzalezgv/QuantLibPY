from abc import ABC, abstractmethod
from typing import Dict

class GreeksStrategy(ABC):
    """Abstract interface for Greeks calculation strategies."""

    @abstractmethod
    def calculate_greeks(self, product, model) -> Dict[str, float]:
        """Return a dict with keys: Price, Delta, Gamma, Theta, Vega, Rho"""
        raise NotImplementedError
