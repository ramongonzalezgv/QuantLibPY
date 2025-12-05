from typing import Dict, Any
import numpy as np

from .StochasticModel import StochasticModel

class BlackModel(StochasticModel):
    """Modelo de Black para forwards."""
    
    def __init__(self, sigma: float, r: float):
        self.sigma = sigma
        self.r = r

    def get_parameters(self) -> Dict[str, Any]:
        """Return model parameters for caching."""
        return {
            'sigma': self.sigma,
            'r': self.r
        }
    
    def characteristic_function(self, u: complex, params: Dict) -> complex:
        """Función característica Black."""
        F = params['F']
        ttm = params['ttm']
        return np.exp(1j * u * np.log(F) - 0.5 * u**2 * self.sigma**2 * ttm)
    
    def d1(self, F: float, K: float, ttm: float) -> float:
        return (np.log(F / K) + 0.5 * self.sigma**2 * ttm) / \
               (self.sigma * np.sqrt(ttm))
    
    def d2(self, F: float, K: float, ttm: float) -> float:
        return self.d1(F, K, ttm) - self.sigma * np.sqrt(ttm)

