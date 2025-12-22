from typing import Optional
from ..products.EuropeanOption import EuropeanOption
from ..products.AsianOption import AsianOption
from ..models.BlackScholesModel import BlackScholesModel
from ..models.HestonModel import HestonModel
from ..models.SABRModel import SABRModel, NormalSABRModel, LognormalSABRModel
import numpy as np



from .PricingEngine import PricingEngine

class MonteCarloEngine(PricingEngine):
    """Monte Carlo engine for stochastic models simulation."""
    
    def __init__(self, n_paths: int = 10000, n_steps: int = 252, seed: Optional[int] = None):
        """
        Parameters
        ----------
        n_paths : int
            Number of simulated Monte Carlo paths
        n_steps : int
            Number of temporal steps in each path
        seed : int, optional
            Seed for reproductibility
        """
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.seed = seed
    
    def calculate_price(self, product, model) -> float:
        """
        Calculate price using Monte Carlo simulation.
        Supports: EuropeanOption y AsianOption.
        """
        ttm = product.ttm
        
        # Determinar si necesitamos la trayectoria completa o solo el precio final
        is_asian = isinstance(product, AsianOption)
        
        if is_asian:
            # Para asiáticas, necesitamos la trayectoria completa
            if isinstance(model, BlackScholesModel):
                # Usar la nueva simulación paso a paso
                S_paths = self._simulate_bs(product.S, model, ttm) 
            elif isinstance(model, HestonModel):
                # Usar la simulación Heston (que ya devuelve la trayectoria)
                S_paths = self._simulate_heston(product.S, model, ttm)
            else:
                raise ValueError(f"Modelo {type(model).__name__} no soportado para AsianOption")

            # Calcular payoff usando la trayectoria completa
            payoffs = product.payoff(S_paths) # product.payoff() manejará el promedio
            
        elif isinstance(product, EuropeanOption):
            # Para europeas, solo necesitamos el precio final (podemos usar la simulación exacta si es BS)
            if isinstance(model, BlackScholesModel):
                # Puedes mantener el atajo log-normal para eficiencia si lo deseas:
                S_T = self._simulate_bs_exact(product.S, model, ttm) # Nueva función auxiliar para el atajo
            elif isinstance(model, HestonModel):
                # Heston debe seguir usando la simulación paso a paso, y tomar solo el último punto
                S_T = self._simulate_heston(product.S, model, ttm)[:, -1] 
            elif isinstance(model, SABRModel):
                # Use SABR Monte Carlo to simulate forward F paths and price using Black/Bachelier depending on beta
                F_T = self._simulate_sabr(product.S if product.F is None else product.F, model, ttm)
                # For pricing, if beta==0 use Bachelier payoff on F; else map to Black-style payoff on forward
                if getattr(model, 'beta', None) == 0 or isinstance(model, NormalSABRModel):
                    # Bachelier payoff
                    vol = model.alpha  # note: alpha is initial vol; simulation already includes vol randomness
                    S_T = F_T
                else:
                    S_T = F_T
            else:
                raise ValueError(f"Model {type(model).__name__} not supported for European Options")
                
            # Calcular payoff usando solo el precio final
            payoffs = product.payoff(S_T)

        else:
            raise ValueError(f"Product type {type(product).__name__} not supported by MonteCarloEngine")

        # Descontar y promediar (el mismo código, ya que 'payoffs' es una matriz 1D en ambos casos)
        discount_factor = np.exp(-model.r * ttm)
        price = discount_factor * np.mean(payoffs)
        
        return price

    def _simulate_sabr(self, F0: float, model: SABRModel, T: float) -> np.ndarray:
        """
        Simulate SABR forward paths (F) using Euler-Maruyama.

        Returns
        -------
        F_T : ndarray of shape (n_paths,)
            Final forward values at time T
        """
        if self.seed is not None:
            np.random.seed(self.seed)

        dt = T / self.n_steps
        n = self.n_steps

        F = np.full((self.n_paths,), F0, dtype=float)
        alpha = np.full((self.n_paths,), model.alpha, dtype=float)

        for i in range(n):
            Z1 = np.random.normal(0.0, 1.0, self.n_paths)
            Z2 = np.random.normal(0.0, 1.0, self.n_paths)
            dW1 = np.sqrt(dt) * Z1
            dW2 = np.sqrt(dt) * (model.rho * Z1 + np.sqrt(max(0.0, 1 - model.rho**2)) * Z2)

            # Forward update: dF = alpha * F^beta * dW1
            F = F + alpha * (F**model.beta) * dW1

            # Volatility update: dalpha = nu * alpha * dW2
            alpha = alpha + model.nu * alpha * dW2
            # Keep alpha positive
            alpha = np.maximum(alpha, 1e-12)

        return F
    
    def _simulate_bs_exact(self, S0: float, model: BlackScholesModel, T: float) -> np.ndarray:
        """
        Simulate paths under Black-Scholes using exact log-normal scheme.
        
        Returns
        -------
        S_T : ndarray of shape (n_paths,)
            Final asset prices at time T
        """
        if self.seed is not None:
            np.random.seed(self.seed)
        
        dt = T / self.n_steps
        
        # Simulación exacta: S_T = S_0 * exp((r - q - 0.5*sigma^2)*T + sigma*sqrt(T)*Z)
        drift = (model.r - model.q - 0.5 * model.sigma**2) * T
        diffusion = model.sigma * np.sqrt(T)
        
        Z = np.random.normal(0, 1, self.n_paths)
        S_T = S0 * np.exp(drift + diffusion * Z)
        
        return S_T
    
    def _simulate_bs(self, S0: float, model: BlackScholesModel, T: float) -> np.ndarray:
        """
        Simulate step by step paths under Black-Scholes using Euler/Log-Euler.
        
        Returns
        -------
        S_paths : ndarray of shape (n_paths, n_steps + 1)
            All asset prices for all paths.
        """
        if self.seed is not None:
            np.random.seed(self.seed)

        dt = T / self.n_steps
        
        # Inicializar arrays
        S = np.zeros((self.n_paths, self.n_steps + 1))
        S[:, 0] = S0

        for n in range(self.n_steps):
            # Generar normales
            Z = np.random.normal(0.0, 1.0, self.n_paths)
            dW = np.sqrt(dt) * Z
            
            # Log-Euler para precio del activo
            S[:, n+1] = S[:, n] * np.exp(
                (model.r - model.q - 0.5 * model.sigma**2) * dt + 
                model.sigma * dW
            )
            
        return S # Retorna toda la matriz de precios (n_paths, n_steps + 1)
    
    def _simulate_heston(self, S0: float, model: HestonModel, T: float) -> np.ndarray:
        """
        Simula trayectorias bajo el modelo de Heston usando esquema de Euler.
        Usa truncación para mantener varianza no-negativa.
        
        Returns
        -------
        S_T : ndarray de forma (n_paths,)
            Precios finales del activo en T
        """
        if self.seed is not None:
            np.random.seed(self.seed)
        
        dt = T / self.n_steps
        
        # Inicializar arrays
        S = np.zeros((self.n_paths, self.n_steps + 1))
        v = np.zeros((self.n_paths, self.n_steps + 1))
        
        S[:, 0] = S0
        v[:, 0] = model.v0
        
        for n in range(self.n_steps):
            # Generar dos conjuntos de normales independientes
            Z1 = np.random.normal(0.0, 1.0, self.n_paths)
            Z2 = np.random.normal(0.0, 1.0, self.n_paths)
            
            # Incrementos correlacionados: dW1 y dW2
            dW1 = np.sqrt(dt) * Z1
            dW2 = np.sqrt(dt) * (model.rho * Z1 + np.sqrt(1 - model.rho**2) * Z2)
            
            # Varianza actual
            v_current = v[:, n]
            
            # Actualización Euler para varianza (truncación para no-negatividad)
            v_next = (v_current + 
                     model.kappa * (model.theta - v_current) * dt + 
                     model.sigma * np.sqrt(np.maximum(v_current, 0.0)) * dW2)
            v_next = np.maximum(v_next, 0.0)
            
            # Actualización log-Euler para precio del activo
            S[:, n+1] = S[:, n] * np.exp(
                (model.r - model.q - 0.5 * v_current) * dt + 
                np.sqrt(np.maximum(v_current, 0.0)) * dW1
            )
            
            v[:, n+1] = v_next
        
        return S  # Retornar solo precios finales