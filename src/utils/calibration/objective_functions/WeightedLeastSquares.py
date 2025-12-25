from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np
from scipy.optimize import minimize, differential_evolution, least_squares
import warnings

from src.utils.dataclasses.MarketQuote import MarketQuote
from src.utils.calibration.objective_functions.ObjectiveFunction import ObjectiveFunction

class WeightedLeastSquares(ObjectiveFunction):
    """
    Weighted least squares objective function.
    
    Minimizes: Σ weight_i * (model_value_i - market_value_i)^2
    """
    
    def __call__(self, params: np.ndarray,
                 market_data: List[MarketQuote],
                 model: Any,
                 param_names: List[str],
                 pricing_func: Callable,
                 **kwargs) -> float:
        """Compute weighted least squares."""
        # Update model parameters
        self._update_model_params(model, params, param_names)
        
        # Compute errors
        total_error = 0.0
        for quote in market_data:
            model_value = pricing_func(model, quote)
            error = (model_value - quote.market_value) ** 2
            total_error += quote.weight * error
        
        return total_error
    
    def _update_model_params(self, model: Any, params: np.ndarray, param_names: List[str]):
        """Update model parameters."""
        for i, name in enumerate(param_names):
            setattr(model, name, params[i])


class RelativeLeastSquares(ObjectiveFunction):
    """
    Relative (percentage) least squares objective function.
    
    Minimizes: Σ weight_i * ((model_value_i - market_value_i) / market_value_i)^2
    
    Useful when market values vary over large ranges (e.g., different maturities).
    """
    
    def __call__(self, params: np.ndarray,
                 market_data: List[MarketQuote],
                 model: Any,
                 param_names: List[str],
                 pricing_func: Callable,
                 **kwargs) -> float:
        """Compute relative least squares."""
        self._update_model_params(model, params, param_names)
        
        total_error = 0.0
        for quote in market_data:
            model_value = pricing_func(model, quote)
            if abs(quote.market_value) > 1e-10:
                relative_error = ((model_value - quote.market_value) / quote.market_value) ** 2
                total_error += quote.weight * relative_error
            else:
                # Fallback to absolute error if market value is too small
                error = (model_value - quote.market_value) ** 2
                total_error += quote.weight * error
        
        return total_error
    
    def _update_model_params(self, model: Any, params: np.ndarray, param_names: List[str]):
        """Update model parameters."""
        for i, name in enumerate(param_names):
            setattr(model, name, params[i])

class VegaWeightedLeastSquares(ObjectiveFunction):
    """
    Vega-weighted least squares objective function.
    
    Weights errors by option vega (sensitivity to volatility).
    Emphasizes liquid, vega-sensitive options in calibration.
    """
    
    def __call__(self, params: np.ndarray,
                 market_data: List[MarketQuote],
                 model: Any,
                 param_names: List[str],
                 pricing_func: Callable,
                 vega_func: Optional[Callable] = None,
                 **kwargs) -> float:
        """Compute vega-weighted least squares."""
        self._update_model_params(model, params, param_names)
        
        total_error = 0.0
        for quote in market_data:
            model_value = pricing_func(model, quote)
            error = (model_value - quote.market_value) ** 2
            
            # Compute vega weight if function provided
            if vega_func is not None:
                vega = vega_func(model, quote)
                weight = quote.weight * abs(vega)
            else:
                weight = quote.weight
            
            total_error += weight * error
        
        return total_error
    
    def _update_model_params(self, model: Any, params: np.ndarray, param_names: List[str]):
        """Update model parameters."""
        for i, name in enumerate(param_names):
            setattr(model, name, params[i])