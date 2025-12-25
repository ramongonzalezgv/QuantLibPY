
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np
from scipy.optimize import minimize, differential_evolution, least_squares
import warnings

from src.utils.dataclasses import MarketQuote, CalibrationResult
from src.utils.calibration.objective_functions import ObjectiveFunction, WeightedLeastSquares

class CalibrationPlotter:
    """
    Visualization tools for model calibration results.
    """
    
    @staticmethod
    def plot_volatility_surface(result: CalibrationResult,
                                 market_data: List[MarketQuote],
                                 model: Any,
                                 pricing_function: Callable,
                                 title: str = "Volatility Surface: Market vs Model",
                                 save_path: Optional[str] = None):
        """
        Plot 3D volatility surface comparing market and model.
        
        Parameters
        ----------
        result : CalibrationResult
            Calibration results
        market_data : List[MarketQuote]
            Market quotes used for calibration
        model : Any
            Calibrated model
        pricing_function : Callable
            Function to compute model values
        title : str
            Plot title
        save_path : str, optional
            Path to save figure
        """
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
        except ImportError:
            print("matplotlib is required for plotting. Install with: pip install matplotlib")
            return
        
        # Extract data
        strikes = np.array([q.strike for q in market_data])
        maturities = np.array([q.maturity for q in market_data])
        market_ivs = np.array([q.market_value for q in market_data])
        
        # Compute model values on grid
        unique_strikes = np.unique(strikes)
        unique_maturities = np.unique(maturities)
        
        K_grid, T_grid = np.meshgrid(unique_strikes, unique_maturities)
        model_grid = np.zeros_like(K_grid)
        
        for i, T in enumerate(unique_maturities):
            for j, K in enumerate(unique_strikes):
                # Find forward for this maturity
                forward = next((q.forward for q in market_data 
                              if abs(q.maturity - T) < 1e-6), None)
                if forward is None:
                    forward = unique_strikes[len(unique_strikes)//2]  # fallback
                
                quote = MarketQuote(K, T, 0.0, "volatility", forward)
                model_grid[i, j] = pricing_function(model, quote)
        
        # Create 3D plot
        fig = plt.figure(figsize=(15, 5))
        
        # Subplot 1: Market surface
        ax1 = fig.add_subplot(131, projection='3d')
        ax1.scatter(strikes, maturities, market_ivs, c='blue', marker='o', s=50, label='Market') #type: ignore
        ax1.set_xlabel('Strike')
        ax1.set_ylabel('Maturity')
        ax1.set_zlabel('Implied Volatility')
        ax1.set_title('Market Volatility Surface')
        ax1.legend()
        
        # Subplot 2: Model surface
        ax2 = fig.add_subplot(132, projection='3d')
        ax2.plot_surface(K_grid, T_grid, model_grid, alpha=0.7, cmap='viridis')
        ax2.scatter(strikes, maturities, result.model_values, c='red', marker='^', s=50, label='Model') #type: ignore
        ax2.set_xlabel('Strike')
        ax2.set_ylabel('Maturity')
        ax2.set_zlabel('Implied Volatility')
        ax2.set_title('Calibrated Model Surface')
        ax2.legend()
        
        # Subplot 3: Comparison
        ax3 = fig.add_subplot(133, projection='3d')
        ax3.scatter(strikes, maturities, market_ivs, c='blue', marker='o', s=50, label='Market', alpha=0.6) #type: ignore
        ax3.scatter(strikes, maturities, result.model_values, c='red', marker='^', s=50, label='Model', alpha=0.6) #type: ignore
        ax3.set_xlabel('Strike')
        ax3.set_ylabel('Maturity')
        ax3.set_zlabel('Implied Volatility')
        ax3.set_title('Market vs Model')
        ax3.legend()
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    @staticmethod
    def plot_volatility_smile(result: CalibrationResult,
                               market_data: List[MarketQuote],
                               model: Any,
                               pricing_function: Callable,
                               title: str = "Volatility Smile: Market vs Model",
                               save_path: Optional[str] = None):
        """
        Plot volatility smiles by maturity (2D curves).
        
        Parameters
        ----------
        result : CalibrationResult
            Calibration results
        market_data : List[MarketQuote]
            Market quotes
        model : Any
            Calibrated model
        pricing_function : Callable
            Function to compute model values
        title : str
            Plot title
        save_path : str, optional
            Path to save figure
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib is required for plotting. Install with: pip install matplotlib")
            return
        
        # Group by maturity
        maturities = sorted(set(q.maturity for q in market_data))
        n_maturities = len(maturities)
        
        # Create subplots
        n_cols = min(3, n_maturities)
        n_rows = (n_maturities + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
        if n_maturities == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        for idx, T in enumerate(maturities):
            ax = axes[idx]
            
            # Filter data for this maturity
            mat_data = [q for q in market_data if abs(q.maturity - T) < 1e-6]
            strikes = np.array([q.strike for q in mat_data])
            market_ivs = np.array([q.market_value for q in mat_data])
            model_ivs = np.array([pricing_function(model, q) for q in mat_data])
            
            # Sort by strike for plotting
            sort_idx = np.argsort(strikes)
            strikes = strikes[sort_idx]
            market_ivs = market_ivs[sort_idx]
            model_ivs = model_ivs[sort_idx]
            
            # Plot
            ax.plot(strikes, market_ivs, 'o-', color='blue', linewidth=2, 
                   markersize=8, label='Market', alpha=0.7)
            ax.plot(strikes, model_ivs, '^--', color='red', linewidth=2, 
                   markersize=8, label='Model', alpha=0.7)
            
            # Mark ATM
            forward = mat_data[0].forward if mat_data[0].forward else strikes[len(strikes)//2]
            ax.axvline(forward, color='gray', linestyle=':', alpha=0.5, label='ATM')
            
            ax.set_xlabel('Strike', fontsize=11)
            ax.set_ylabel('Implied Volatility', fontsize=11)
            ax.set_title(f'Maturity = {T:.2f}Y', fontsize=12, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(n_maturities, len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    @staticmethod
    def plot_error_analysis(result: CalibrationResult,
                            market_data: List[MarketQuote],
                            title: str = "Calibration Error Analysis",
                            save_path: Optional[str] = None):
        """
        Plot error analysis: errors by strike, maturity, and histogram.
        
        Parameters
        ----------
        result : CalibrationResult
            Calibration results
        market_data : List[MarketQuote]
            Market quotes
        title : str
            Plot title
        save_path : str, optional
            Path to save figure
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib is required for plotting. Install with: pip install matplotlib")
            return
        
        strikes = np.array([q.strike for q in market_data])
        maturities = np.array([q.maturity for q in market_data])
        errors = result.errors
        relative_errors = errors / result.market_values * 100  # percentage
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Errors vs Strike
        ax1 = axes[0, 0]
        scatter1 = ax1.scatter(strikes, errors, c=maturities, cmap='viridis', s=100, alpha=0.6)
        ax1.axhline(0, color='red', linestyle='--', alpha=0.5)
        ax1.set_xlabel('Strike', fontsize=11)
        ax1.set_ylabel('Error (Model - Market)', fontsize=11)
        ax1.set_title('Errors vs Strike', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        plt.colorbar(scatter1, ax=ax1, label='Maturity')
        
        # 2. Errors vs Maturity
        ax2 = axes[0, 1]
        unique_maturities = sorted(set(maturities))
        avg_errors = [np.mean(np.abs(errors[maturities == T])) for T in unique_maturities]
        ax2.bar(unique_maturities, avg_errors, color='steelblue', alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Maturity', fontsize=11)
        ax2.set_ylabel('Average Absolute Error', fontsize=11)
        ax2.set_title('Average Error by Maturity', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. Error Histogram
        ax3 = axes[1, 0]
        ax3.hist(relative_errors, bins=20, color='coral', alpha=0.7, edgecolor='black')
        ax3.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
        ax3.set_xlabel('Relative Error (%)', fontsize=11)
        ax3.set_ylabel('Frequency', fontsize=11)
        ax3.set_title('Error Distribution', fontsize=12, fontweight='bold')
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. Market vs Model scatter
        ax4 = axes[1, 1]
        ax4.scatter(result.market_values, result.model_values, c=maturities, 
                   cmap='viridis', s=100, alpha=0.6)
        
        # Perfect fit line
        min_val = min(result.market_values.min(), result.model_values.min())
        max_val = max(result.market_values.max(), result.model_values.max())
        ax4.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, 
                label='Perfect Fit', alpha=0.7)
        
        ax4.set_xlabel('Market Value', fontsize=11)
        ax4.set_ylabel('Model Value', fontsize=11)
        ax4.set_title('Market vs Model Values', fontsize=12, fontweight='bold')
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3)
        ax4.set_aspect('equal', adjustable='box')
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    @staticmethod
    def plot_calibration_summary(result: CalibrationResult,
                                  market_data: List[MarketQuote],
                                  model: Any,
                                  pricing_function: Callable,
                                  title: str = "Calibration Summary",
                                  save_path: Optional[str] = None):
        """
        Create comprehensive calibration summary plot with all key visualizations.
        
        Parameters
        ----------
        result : CalibrationResult
            Calibration results
        market_data : List[MarketQuote]
            Market quotes
        model : Any
            Calibrated model
        pricing_function : Callable
            Function to compute model values
        title : str
            Plot title
        save_path : str, optional
            Path to save figure
        """
        try:
            import matplotlib.pyplot as plt
            from matplotlib.gridspec import GridSpec
        except ImportError:
            print("matplotlib is required for plotting. Install with: pip install matplotlib")
            return
        
        fig = plt.figure(figsize=(18, 12))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        strikes = np.array([q.strike for q in market_data])
        maturities = np.array([q.maturity for q in market_data])
        errors = result.errors
        relative_errors = errors / result.market_values * 100
        
        # 1. Volatility smiles (top row, spanning 2 columns)
        maturities_unique = sorted(set(maturities))
        n_plots = min(3, len(maturities_unique))
        
        for i in range(n_plots):
            ax = fig.add_subplot(gs[0, i])
            T = maturities_unique[i]
            
            mat_data = [q for q in market_data if abs(q.maturity - T) < 1e-6]
            K = np.array([q.strike for q in mat_data])
            market_iv = np.array([q.market_value for q in mat_data])
            model_iv = np.array([pricing_function(model, q) for q in mat_data])
            
            sort_idx = np.argsort(K)
            ax.plot(K[sort_idx], market_iv[sort_idx], 'o-', color='blue', 
                   linewidth=2, markersize=8, label='Market', alpha=0.7)
            ax.plot(K[sort_idx], model_iv[sort_idx], '^--', color='red', 
                   linewidth=2, markersize=8, label='Model', alpha=0.7)
            
            ax.set_xlabel('Strike')
            ax.set_ylabel('Implied Volatility')
            ax.set_title(f'T = {T:.2f}Y', fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 2. Error scatter (middle left)
        ax2 = fig.add_subplot(gs[1, 0])
        scatter = ax2.scatter(strikes, errors, c=maturities, cmap='viridis', 
                             s=100, alpha=0.6, edgecolors='black')
        ax2.axhline(0, color='red', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Strike')
        ax2.set_ylabel('Error')
        ax2.set_title('Errors vs Strike', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax2, label='Maturity')
        
        # 3. Error histogram (middle center)
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.hist(relative_errors, bins=20, color='coral', alpha=0.7, edgecolor='black')
        ax3.axvline(0, color='red', linestyle='--', linewidth=2)
        ax3.set_xlabel('Relative Error (%)')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Error Distribution', fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. Market vs Model (middle right)
        ax4 = fig.add_subplot(gs[1, 2])
        ax4.scatter(result.market_values, result.model_values, c=maturities,
                   cmap='viridis', s=100, alpha=0.6, edgecolors='black')
        min_val = min(result.market_values.min(), result.model_values.min())
        max_val = max(result.market_values.max(), result.model_values.max())
        ax4.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, alpha=0.7)
        ax4.set_xlabel('Market Value')
        ax4.set_ylabel('Model Value')
        ax4.set_title('Market vs Model', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.set_aspect('equal', adjustable='box')
        
        # 5. Calibrated parameters table (bottom left)
        ax5 = fig.add_subplot(gs[2, 0])
        ax5.axis('off')
        
        param_text = "Calibrated Parameters:\n" + "="*30 + "\n"
        for param, value in result.calibrated_params.items():
            param_text += f"{param:>10} = {value:>10.6f}\n"
        
        ax5.text(0.1, 0.5, param_text, fontsize=12, family='monospace',
                verticalalignment='center', bbox=dict(boxstyle='round', 
                facecolor='lightblue', alpha=0.5))
        
        # 6. Statistics table (bottom center)
        ax6 = fig.add_subplot(gs[2, 1])
        ax6.axis('off')
        
        stats_text = "Fit Statistics:\n" + "="*30 + "\n"
        stats_text += f"{'RMSE':<20} = {result.rmse:>10.6f}\n"
        stats_text += f"{'MAPE':<20} = {result.mape:>9.2%}\n"
        stats_text += f"{'Max Error':<20} = {result.max_error:>10.6f}\n"
        stats_text += f"{'Objective':<20} = {result.objective_value:>10.6e}\n"
        stats_text += f"{'Iterations':<20} = {result.iterations:>10d}\n"
        
        ax6.text(0.1, 0.5, stats_text, fontsize=12, family='monospace',
                verticalalignment='center', bbox=dict(boxstyle='round',
                facecolor='lightgreen', alpha=0.5))
        
        # 7. Error by maturity bar chart (bottom right)
        ax7 = fig.add_subplot(gs[2, 2])
        unique_mats = sorted(set(maturities))
        avg_errors = [np.mean(np.abs(errors[maturities == T])) for T in unique_mats]
        ax7.bar(range(len(unique_mats)), avg_errors, color='steelblue', 
               alpha=0.7, edgecolor='black')
        ax7.set_xticks(range(len(unique_mats)))
        ax7.set_xticklabels([f'{T:.1f}Y' for T in unique_mats])
        ax7.set_ylabel('Avg Absolute Error')
        ax7.set_title('Error by Maturity', fontweight='bold')
        ax7.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()