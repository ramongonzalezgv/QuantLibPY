from typing import Dict, Any
import numpy as np

from .StochasticModel import StochasticModel

class BlackScholesModel(StochasticModel):
    """Modelo Black-Scholes clásico."""
    
    def __init__(self, sigma: float, r: float, q: float = 0):
        self.sigma = sigma
        self.r = r
        self.q = q

    def get_parameters(self) -> Dict[str, Any]:
        """Return model parameters for caching."""
        return {
            'sigma': self.sigma,
            'r': self.r,
            'q': self.q
        }
    
    def characteristic_function(self, u: complex, params: Dict) -> complex:
        """Función característica BS con dividendos."""
        ttm = params['ttm']
        return np.exp(
            1j * u * (self.r - self.q - 0.5 * self.sigma**2) * ttm 
            - 0.5 * u**2 * (self.sigma * np.sqrt(ttm))**2
        )
    
    def d1(self, S: float, K: float, ttm: float) -> float:
        return (np.log(S / K) + (self.r - self.q + 0.5 * self.sigma**2) * ttm) / \
               (self.sigma * np.sqrt(ttm))
    
    def d2(self, S: float, K: float, ttm: float) -> float:
        return self.d1(S, K, ttm) - self.sigma * np.sqrt(ttm)
