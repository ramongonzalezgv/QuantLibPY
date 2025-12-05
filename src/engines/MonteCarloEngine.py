from typing import Optional
from ..products.EuropeanOption import EuropeanOption
from ..products.AsianOption import AsianOption
from ..models.BlackScholesModel import BlackScholesModel
from ..models.HestonModel import HestonModel
import numpy as np



from .PricingEngine import PricingEngine

class MonteCarloEngine(PricingEngine):
    """Motor de Monte Carlo para simulación de modelos estocásticos."""
    
    def __init__(self, n_paths: int = 10000, n_steps: int = 252, seed: Optional[int] = None):
        """
        Parameters
        ----------
        n_paths : int
            Número de trayectorias de Monte Carlo
        n_steps : int
            Número de pasos temporales por trayectoria
        seed : int, optional
            Semilla para reproducibilidad
        """
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.seed = seed
    
    def calculate_price(self, product, model) -> float:
        """
        Calcula el precio usando simulación Monte Carlo.
        Soporta: EuropeanOption y AsianOption.
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
            else:
                raise ValueError(f"Modelo {type(model).__name__} no soportado para EuropeanOption")
                
            # Calcular payoff usando solo el precio final
            payoffs = product.payoff(S_T)

        else:
            raise ValueError(f"Tipo de producto {type(product).__name__} no soportado por MonteCarloEngine")

        # Descontar y promediar (el mismo código, ya que 'payoffs' es una matriz 1D en ambos casos)
        discount_factor = np.exp(-model.r * ttm)
        price = discount_factor * np.mean(payoffs)
        
        return price
        
    # def calculate_price(self, product, model) -> float:
    #     """
    #     Calcula el precio usando simulación Monte Carlo.
    #     Soporta: BlackScholesModel y HestonModel.
    #     """
    #     if not isinstance(product, EuropeanOption):
    #         raise ValueError("MonteCarloEngine actualmente solo soporta EuropeanOption")
        
    #     ttm = product.ttm
        
    #     if isinstance(model, BlackScholesModel):
    #         S_T = self._simulate_bs_exact(product.S, model, ttm)
    #     elif isinstance(model, HestonModel):
    #         S = self._simulate_heston(product.S, model, ttm)
    #         S_T = S[:, -1]
    #     else:
    #         raise ValueError(f"Modelo {type(model).__name__} no soportado por MonteCarloEngine")
        
    #     # Calcular payoff en todos los paths
    #     payoffs = product.payoff(S_T)
        
    #     # Descontar y promediar
    #     discount_factor = np.exp(-model.r * ttm)
    #     price = discount_factor * np.mean(payoffs)
        
    #     return price
    
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