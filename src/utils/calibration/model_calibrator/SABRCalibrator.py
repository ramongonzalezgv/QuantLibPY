
from typing import List, Tuple, Optional

from .ModelCalibrator import ModelCalibrator
from src.utils.calibration.objective_functions import RelativeLeastSquares
from src.utils.dataclasses import MarketQuote

class SABRCalibrator(ModelCalibrator):
    """
    Specialized calibrator for SABR models.
    
    Provides convenient interface for SABR calibration with sensible defaults.
    """
    
    def __init__(self,
                 model,
                 param_names: Optional[List[str]] = None,
                 bounds: Optional[List[Tuple[float, float]]] = None,
                 method: str = "L-BFGS-B"):
        """
        Initialize SABR calibrator.
        
        Parameters
        ----------
        model : SABRModel
            SABR model instance
        param_names : List[str], optional
            Parameters to calibrate (default: ['alpha', 'rho', 'nu'])
        bounds : List[Tuple[float, float]], optional
            Parameter bounds (uses sensible defaults if not provided)
        method : str
            Optimization method
        """
        # Default parameters and bounds for SABR
        if param_names is None:
            param_names = ['alpha', 'rho', 'nu']
        
        if bounds is None:
            bounds = [
                (0.001, 2.0),   # alpha
                (-0.999, 0.999), # rho
                (0.001, 2.0)     # nu
            ]
        
        # Pricing function for SABR (returns implied volatility)
        def sabr_pricing_function(model, quote: MarketQuote) -> float:
            if quote.forward is None:
                raise ValueError("MarketQuote must have 'forward' set for SABR calibration")
            return model.implied_volatility(quote.forward, quote.strike, quote.maturity)
        
        super().__init__(
            model=model,
            param_names=param_names,
            bounds=bounds,
            pricing_function=sabr_pricing_function,
            objective=RelativeLeastSquares(),  # Use relative LS for volatilities
            method=method
        )
