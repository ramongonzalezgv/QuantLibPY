from typing import Dict, Any
import numpy as np

from .StochasticModel import StochasticModel

class HestonModel(StochasticModel):
    """Modelo de Heston con volatilidad estocástica."""
    
    def __init__(self, kappa: float, theta: float, sigma: float, 
                 rho: float, v0: float, r: float, q: float = 0):
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.rho = rho
        self.v0 = v0
        self.r = r
        self.q = q
    
    def get_parameters(self) -> Dict[str, Any]:
        """Return model parameters for caching."""
        return {
            'kappa': self.kappa,
            'theta': self.theta,
            'sigma': self.sigma,
            'rho': self.rho,
            'v0': self.v0,
            'r': self.r,
            'q': self.q
        }
    
    def characteristic_function(self, u: complex, params: Dict) -> complex:
        """Función característica de Heston."""
        ttm = params['ttm']
        
        d = np.sqrt(
            (self.rho * self.sigma * u * 1j - self.kappa)**2 + 
            (self.sigma**2) * (u * 1j + u**2)
        )
        g = (self.kappa - self.rho * self.sigma * u * 1j - d) / \
            (self.kappa - self.rho * self.sigma * u * 1j + d)
        
        exp_dt = np.exp(-d * ttm)
        G = (1 - g * exp_dt) / (1 - g)
        
        C = (self.r - self.q) * u * 1j * ttm + \
            (self.kappa * self.theta / self.sigma**2) * \
            ((self.kappa - self.rho * self.sigma * u * 1j - d) * ttm - 2 * np.log(G))
        D = ((self.kappa - self.rho * self.sigma * u * 1j - d) / self.sigma**2) * \
            ((1 - exp_dt) / (1 - g * exp_dt))
        
        return np.exp(C + D * self.v0)
    