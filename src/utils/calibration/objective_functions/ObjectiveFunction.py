from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np
from scipy.optimize import minimize, differential_evolution, least_squares
import warnings

from src.utils.dataclasses.MarketQuote import MarketQuote

class ObjectiveFunction(ABC):
    """
    Abstract base class for calibration objective functions.
    """
    
    @abstractmethod
    def __call__(self, params: np.ndarray, 
                 market_data: List[MarketQuote],
                 model: Any,
                 **kwargs) -> float:
        """
        Compute the objective function value.
        
        Parameters
        ----------
        params : np.ndarray
            Model parameters to optimize
        market_data : List[MarketQuote]
            Market quotes for calibration
        model : Any
            The model being calibrated
        """
        pass