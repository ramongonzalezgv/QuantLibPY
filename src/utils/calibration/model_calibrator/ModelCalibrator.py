from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np
from scipy.optimize import minimize, differential_evolution, least_squares
import warnings

from src.utils.calibration.objective_functions import ObjectiveFunction, WeightedLeastSquares
from src.utils.dataclasses import MarketQuote, CalibrationResult
from src.utils.calibration.plotting import CalibrationPlotter

class ModelCalibrator:
    """
    Generic model calibration engine.
    
    Supports multiple optimization algorithms and objective functions.
    """
    
    def __init__(self,
                 model: Any,
                 param_names: List[str],
                 bounds: List[Tuple[float, float]],
                 pricing_function: Callable,
                 objective: Optional[ObjectiveFunction] = None,
                 method: str = "L-BFGS-B"):
        """
        Initialize calibrator.
        
        Parameters
        ----------
        model : Any
            Model instance to calibrate (SABR, Heston, etc.)
        param_names : List[str]
            Names of parameters to calibrate (e.g., ['alpha', 'rho', 'nu'])
        bounds : List[Tuple[float, float]]
            Parameter bounds [(min, max), ...]
        pricing_function : Callable
            Function that computes model values: pricing_function(model, quote) -> float
        objective : ObjectiveFunction, optional
            Objective function to minimize (default: WeightedLeastSquares)
        method : str
            Optimization method: 'L-BFGS-B', 'SLSQP', 'differential_evolution', etc.
        """
        self.model = model
        self.param_names = param_names
        self.bounds = bounds
        self.pricing_function = pricing_function
        self.objective = objective or WeightedLeastSquares()
        self.method = method
        
        # Store last calibration for plotting
        self._last_result = None
        self._last_market_data = None
        
        # Validate
        if len(param_names) != len(bounds):
            raise ValueError("Number of parameter names must match number of bounds")
    
    def calibrate(self,
                  market_data: List[MarketQuote],
                  initial_guess: Optional[np.ndarray] = None,
                  constraints: Optional[List] = None,
                  **optimizer_kwargs) -> CalibrationResult:
        """
        Calibrate model to market data.
        
        Parameters
        ----------
        market_data : List[MarketQuote]
            Market quotes to calibrate to
        initial_guess : np.ndarray, optional
            Initial parameter guess (default: mid-point of bounds)
        constraints : List, optional
            Additional constraints for optimization
        **optimizer_kwargs
            Additional arguments passed to optimizer
        
        Returns
        -------
        CalibrationResult
            Calibration results including calibrated parameters and fit statistics
        """
        # Set initial guess
        if initial_guess is None:
            initial_guess = np.array([(b[0] + b[1]) / 2 for b in self.bounds])
        
        # Store original parameters for restoration if needed
        original_params = {name: getattr(self.model, name) for name in self.param_names}
        
        try:
            # Run optimization
            if self.method == "differential_evolution":
                result = self._optimize_differential_evolution(
                    market_data, initial_guess, **optimizer_kwargs
                )
            elif self.method == "least_squares":
                result = self._optimize_least_squares(
                    market_data, initial_guess, **optimizer_kwargs
                )
            else:
                result = self._optimize_scipy(
                    market_data, initial_guess, constraints, **optimizer_kwargs
                )
            
            # Compute final errors and statistics
            calibrated_params = {name: result.x[i] for i, name in enumerate(self.param_names)}
            self._update_model_params(result.x)
            
            errors, market_vals, model_vals = self._compute_errors(market_data)
            
            rmse = np.sqrt(np.mean(errors**2))
            mape = np.mean(np.abs(errors / market_vals)) if np.all(market_vals > 1e-10) else np.inf
            max_error = np.max(np.abs(errors))
            
            calib_result = CalibrationResult(
                success=result.success,
                calibrated_params=calibrated_params,
                objective_value=result.fun,
                errors=errors,
                market_values=market_vals,
                model_values=model_vals,
                rmse=rmse,
                mape=mape, #type: ignore
                max_error=max_error,
                iterations=result.nit if hasattr(result, 'nit') else result.nfev,
                message=result.message
            )
            
            # Store for plotting
            self._last_result = calib_result
            self._last_market_data = market_data
            
            return calib_result
        
        except Exception as e:
            # Restore original parameters
            for name, value in original_params.items():
                setattr(self.model, name, value)
            
            raise RuntimeError(f"Calibration failed: {str(e)}")
    
    def plot_results(self, plot_type: str = "summary", save_path: Optional[str] = None):
        """
        Plot calibration results.
        
        Parameters
        ----------
        plot_type : str
            Type of plot: 'summary', 'surface', 'smile', or 'errors'
        save_path : str, optional
            Path to save figure
        """
        if self._last_result is None or self._last_market_data is None:
            raise ValueError("No calibration results available. Run calibrate() first.")
        
        plotter = CalibrationPlotter()
        
        if plot_type == "summary":
            plotter.plot_calibration_summary(
                self._last_result, self._last_market_data, 
                self.model, self.pricing_function, save_path=save_path
            )
        elif plot_type == "surface":
            plotter.plot_volatility_surface(
                self._last_result, self._last_market_data,
                self.model, self.pricing_function, save_path=save_path
            )
        elif plot_type == "smile":
            plotter.plot_volatility_smile(
                self._last_result, self._last_market_data,
                self.model, self.pricing_function, save_path=save_path
            )
        elif plot_type == "errors":
            plotter.plot_error_analysis(
                self._last_result, self._last_market_data, save_path=save_path
            )
        else:
            raise ValueError(f"Unknown plot type: {plot_type}")
    
    def _optimize_scipy(self, market_data, initial_guess, constraints, **kwargs):
        """Optimize using scipy.optimize.minimize."""
        return minimize(
            fun=self.objective,
            x0=initial_guess,
            args=(market_data, self.model, self.param_names, self.pricing_function),
            method=self.method,
            bounds=self.bounds,
            constraints=constraints,
            **kwargs
        )
    
    def _optimize_differential_evolution(self, market_data, initial_guess, **kwargs):
        """Optimize using differential evolution (global optimizer)."""
        default_kwargs = {
            'maxiter': 1000,
            'popsize': 15,
            'tol': 1e-7,
            'atol': 1e-7,
            'polish': True,
            'seed': 42
        }
        default_kwargs.update(kwargs)
        
        return differential_evolution(
            func=self.objective,
            bounds=self.bounds,
            args=(market_data, self.model, self.param_names, self.pricing_function),
            **default_kwargs
        )
    
    def _optimize_least_squares(self, market_data, initial_guess, **kwargs):
        """Optimize using scipy.optimize.least_squares (for residual-based objectives)."""
        def residuals(params):
            self._update_model_params(params)
            res = []
            for quote in market_data:
                model_value = self.pricing_function(self.model, quote)
                error = model_value - quote.market_value
                res.append(np.sqrt(quote.weight) * error)
            return np.array(res)
        
        default_kwargs = {
            'method': 'trf',
            'ftol': 1e-8,
            'xtol': 1e-8,
            'max_nfev': 1000
        }
        default_kwargs.update(kwargs)
        
        return least_squares(
            fun=residuals,
            x0=initial_guess,
            bounds=([b[0] for b in self.bounds], [b[1] for b in self.bounds]),
            **default_kwargs
        )
    
    def _update_model_params(self, params: np.ndarray):
        """Update model parameters."""
        for i, name in enumerate(self.param_names):
            setattr(self.model, name, params[i])
    
    def _compute_errors(self, market_data: List[MarketQuote]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute errors between model and market."""
        errors = []
        market_vals = []
        model_vals = []
        
        for quote in market_data:
            model_value = self.pricing_function(self.model, quote)
            error = model_value - quote.market_value
            
            errors.append(error)
            market_vals.append(quote.market_value)
            model_vals.append(model_value)
        
        return np.array(errors), np.array(market_vals), np.array(model_vals)

