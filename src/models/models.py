from abc import ABC, abstractmethod
from typing import Dict, Any
import numpy as np
import scipy.stats as si
from ..products.products import EuropeanOption


# ============================================================================
# MODELS: Definen el comportamiento estocástico del subyacente
# ============================================================================

class StochasticModel(ABC):
    """Clase base para modelos estocásticos."""
    
    @abstractmethod
    def characteristic_function(self, u: complex, params: Dict) -> complex:
        """Función característica del modelo."""
        pass

    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """
        Método obligatorio para devolver los parámetros de mercado (volatilidad, 
        tasas, etc.) para la clave de caché.
        """
        pass


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
    

# ============================================================================
# GREEKS CALCULATOR: Calcula las griegas independientemente
# ============================================================================

class GreeksCalculator:
    """Calcula las griegas de opciones."""
    
    @staticmethod
    def delta(product: EuropeanOption, model: BlackScholesModel) -> float:
        """Delta de la opción."""
        d1 = model.d1(product.S, product.K, product.ttm)
        
        if product.option_type == "call":
            delta = np.exp(-model.q * product.ttm) * si.norm.cdf(d1)
        else:
            delta = np.exp(-model.q * product.ttm) * (si.norm.cdf(d1) - 1)
        
        return delta * product.qty
    
    @staticmethod
    def gamma(product: EuropeanOption, model: BlackScholesModel) -> float:
        """Gamma de la opción."""
        d1 = model.d1(product.S, product.K, product.ttm)
        gamma = (np.exp(-model.q * product.ttm) * si.norm.pdf(d1)) / \
                (product.S * model.sigma * np.sqrt(product.ttm))
        return gamma * product.qty
    
    @staticmethod
    def vega(product: EuropeanOption, model: BlackScholesModel) -> float:
        """Vega de la opción."""
        d1 = model.d1(product.S, product.K, product.ttm)
        vega = product.S * np.exp(-model.q * product.ttm) * \
               si.norm.pdf(d1) * np.sqrt(product.ttm) / 100
        return vega * product.qty
    
    @staticmethod
    def theta(product: EuropeanOption, model: BlackScholesModel) -> float:
        """Theta de la opción."""
        d1 = model.d1(product.S, product.K, product.ttm)
        d2 = model.d2(product.S, product.K, product.ttm)
        
        first_term = (-product.S * si.norm.pdf(d1) * model.sigma * 
                     np.exp(-model.q * product.ttm)) / (2 * np.sqrt(product.ttm))
        second_term = model.r * product.K * np.exp(-model.r * product.ttm) * \
                     si.norm.cdf(d2 if product.option_type == "call" else -d2)
        third_term = model.q * product.S * np.exp(-model.q * product.ttm) * \
                    si.norm.cdf(d1 if product.option_type == "call" else -d1)
        
        if product.option_type == "call":
            theta = (first_term - second_term - third_term) / 365
        else:
            theta = (first_term + second_term - third_term) / 365
        
        return theta * product.qty
    
    @staticmethod
    def rho(product: EuropeanOption, model: BlackScholesModel) -> float:
        """Rho de la opción."""
        d2 = model.d2(product.S, product.K, product.ttm)
        
        if product.option_type == "call":
            rho = (product.K * product.ttm * np.exp(-model.r * product.ttm) * 
                  si.norm.cdf(d2)) / 100
        else:
            rho = (-product.K * product.ttm * np.exp(-model.r * product.ttm) * 
                  si.norm.cdf(-d2)) / 100
        
        return rho * product.qty
    
    @staticmethod
    def all_greeks(product: EuropeanOption, model: BlackScholesModel) -> Dict[str, float]:
        """Calcula todas las griegas."""
        return {
            'Delta': GreeksCalculator.delta(product, model),
            'Gamma': GreeksCalculator.gamma(product, model),
            'Vega': GreeksCalculator.vega(product, model),
            'Theta': GreeksCalculator.theta(product, model),
            'Rho': GreeksCalculator.rho(product, model)
        }
