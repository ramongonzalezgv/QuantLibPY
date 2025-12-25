
from typing import Dict
from dataclasses import dataclass
import numpy as np


@dataclass
class CalibrationResult:
    """
    Stores the result of a calibration procedure.
    """
    success: bool
    calibrated_params: Dict[str, float]
    objective_value: float
    errors: np.ndarray
    market_values: np.ndarray
    model_values: np.ndarray
    rmse: float
    mape: float  # Mean Absolute Percentage Error
    max_error: float
    iterations: int
    message: str
    
    def summary(self) -> str:
        """Return a formatted summary of calibration results."""
        summary = f"""
        Calibration Results
        {'='*60}
        Success: {self.success}
        Message: {self.message}
        Iterations: {self.iterations}
        
        Objective Value: {self.objective_value:.6e}
        RMSE: {self.rmse:.6e}
        MAPE: {self.mape:.2%}
        Max Error: {self.max_error:.6e}
        
        Calibrated Parameters:
        {'-'*60}
        """
        for param, value in self.calibrated_params.items():
            summary += f"  {param:<15} = {value:>12.6f}\n"
        
        return summary
    
    def get_errors_by_maturity(self, maturities: np.ndarray) -> Dict[float, float]:
        """Calculate average errors grouped by maturity."""
        unique_maturities = np.unique(maturities)
        errors_by_mat = {}
        
        for mat in unique_maturities:
            mask = maturities == mat
            errors_by_mat[mat] = np.mean(np.abs(self.errors[mask]))
        
        return errors_by_mat