from typing import Dict, Any, Optional
import numpy as np
from abc import ABC, abstractmethod

class SABRModel(ABC):
    """
    Abstract base class for SABR models.

    Defines:
    - shared parameters (alpha, rho, nu, r, F0)
    - shared SABR utilities
    - abstract interface for beta and implied volatility
    """

    def __init__(
        self,
        alpha: float,
        rho: float,
        nu: float,
        r: float,
        F0: Optional[float] = None,
    ):
        """
        Initialize SABR model.
        
        Parameters
        ----------
        alpha : float
            Initial volatility (α > 0)
        beta : float
            CEV exponent (0 ≤ β ≤ 1)
            - β = 0: Normal SABR
            - β = 1: Lognormal SABR
        rho : float
            Correlation between forward and volatility (-1 ≤ ρ ≤ 1)
        nu : float
            Volatility of volatility (ν > 0)
        r : float
            Risk-free interest rate
        F0 : float, optional
            Initial forward price. If None, must be provided in params dict.
        """
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if not (-1.0 <= rho <= 1.0):
            raise ValueError("rho must be between -1 and 1")
        if nu <= 0:
            raise ValueError("nu must be positive")

        self.alpha = alpha
        self.rho = rho
        self.nu = nu
        self.r = r
        self.F0 = F0

    @property
    @abstractmethod
    def beta(self) -> float:
        """
        SABR beta parameter.

        Implemented by concrete subclasses:
        - 0.0 for Normal SABR
        - 1.0 for Lognormal SABR
        - (0,1) for General SABR
        """
        pass

    def get_parameters(self) -> Dict[str, Any]:
        params = {
            "alpha": self.alpha,
            "beta": self.beta,   # <-- works polymorphically
            "rho": self.rho,
            "nu": self.nu,
            "r": self.r,
        }
        if self.F0 is not None:
            params["F0"] = self.F0
        return params
    
    def _compute_chi(self, z: float) -> float:
        discriminant = 1.0 - 2.0 * self.rho * z + z * z
        sqrt_term = np.sqrt(max(0.0, discriminant))

        numerator = sqrt_term + z - self.rho
        denominator = 1.0 - self.rho

        if abs(denominator) < 1e-12:
            return z

        if numerator <= 0.0:
            return abs(z)

        return np.log(numerator / denominator)
    
    @abstractmethod
    def implied_volatility(self, F: float, K: float, ttm: float) -> float:
        """
        Compute implied volatility using the appropriate SABR approximation.
        """
        pass


class GeneralSABRModel(SABRModel):
    """
    SABR (Stochastic Alpha Beta Rho) Model.
    
    The SABR model is a stochastic volatility model where both the forward price
    and its volatility follow stochastic processes:
    
    dF = α * F^β * dW1
    dα = ν * α * dW2
    dW1 * dW2 = ρ * dt
    
    where:
    - F: forward price
    - α: volatility (stochastic)
    - β: CEV exponent (controls the relationship between price and volatility)
    - ν: volatility of volatility
    - ρ: correlation between forward and volatility processes
    
    Reference: Hagan, P. S., Kumar, D., Lesniewski, A. S., & Woodward, D. E. (2002).
    "Managing smile risk." The Best of Wilmott, 1, 249-296.
    
    Note: The SABR model does not have a closed-form characteristic function.
    For FFT-based pricing, Monte Carlo simulation or numerical methods are required.
    This class does not inherit from StochasticModel since it cannot provide
    a characteristic function. Use NormalSABRModel or LognormalSABRModel for
    specific implementations.
    """

    def __init__(self, alpha, beta, rho, nu, r, F0=None):
        if not (0.0 < beta < 1.0):
            raise ValueError("beta must be in (0,1) for General SABR")

        super().__init__(alpha, rho, nu, r, F0)
        self._beta = beta

    @property
    def beta(self) -> float:
        return self._beta
    
    def implied_volatility(self, F: float, K: float, ttm: float) -> float:
        """
        Compute implied volatility using Hagan's 2002 approximation.
        
        Parameters
        ----------
        F : float
            Current forward price
        K : float
            Strike price
        ttm : float
            Time to maturity
        
        Returns
        -------
        float
            Implied volatility (Black volatility)
        """
        if F <= 0 or K <= 0:
            raise ValueError("Forward and strike prices must be positive")
        if ttm <= 0:
            raise ValueError("Time to maturity must be positive")
        
        # Handle at-the-money case
        if abs(F - K) < 1e-10:
            return self._atm_volatility(F, ttm)
        
        # Compute z and chi(z)
        z = self._compute_z(F, K)
        chi_z = self._compute_chi(z)
        
        if abs(chi_z) < 1e-10:
            # Fallback to ATM volatility for very small chi
            return self._atm_volatility(F, ttm)
        
        # Main term
        main_term = self.alpha / ((F * K)**((1 - self.beta) / 2)) * z / chi_z
        
        # Correction terms
        correction = self._compute_correction(F, K, ttm, z, chi_z)
        
        return main_term * (1 + correction)
    
    def _compute_z(self, F: float, K: float) -> float:
        """
        Compute z parameter.
        
        For β > 0, uses: z = (ν/α) * (FK)^((1-β)/2) * ln(F/K)
        For β = 0, uses: z = (ν/α) * (F - K) (handled in NormalSABRModel)
        """
        if self.beta == 0:
            # Normal case - should use specialized NormalSABRModel
            return self.nu / self.alpha * (F - K)
        else:
            # General case: z = (ν/α) * (FK)^((1-β)/2) * ln(F/K)
            return (self.nu / self.alpha * 
                   (F * K)**((1 - self.beta) / 2) * np.log(F / K))
    
    def _compute_chi(self, z: float) -> float:
        """Compute chi(z) = ln((sqrt(1 - 2*rho*z + z^2) + z - rho) / (1 - rho))."""
        discriminant = 1 - 2 * self.rho * z + z**2
        if discriminant < 0:
            # Handle numerical issues
            discriminant = max(0, discriminant)
        
        sqrt_term = np.sqrt(discriminant)
        numerator = sqrt_term + z - self.rho
        denominator = 1 - self.rho
        
        if abs(denominator) < 1e-10:
            # Handle rho = 1 case
            return z
        
        if numerator <= 0:
            # Handle edge cases
            return abs(z)
        
        return np.log(numerator / denominator)
    
    def _compute_correction(self, F: float, K: float, ttm: float, 
                           z: float, chi_z: float) -> float:
        """
        Compute correction terms in Hagan's formula.
        
        This includes the time-dependent correction and higher-order terms.
        """
        # First-order correction
        log_FK = np.log(F / K)
        FK_power = (F * K)**((1 - self.beta) / 2)
        
        # Terms from Hagan's expansion
        term1 = ((1 - self.beta)**2 / 24 * (self.alpha**2) / (FK_power**2) +
                 self.rho * self.beta * self.nu * self.alpha / (4 * FK_power) +
                 (2 - 3 * self.rho**2) * self.nu**2 / 24) * ttm
        
        # Additional correction for small strikes
        if abs(z) > 1e-6:
            term2 = ((1 - self.beta)**2 * (log_FK**2) / 24 +
                     (1 - self.beta)**4 * (log_FK**4) / 1920)
        else:
            term2 = 0
        
        return term1 + term2
    
    def _atm_volatility(self, F: float, ttm: float) -> float:
        """
        Compute at-the-money implied volatility.
        
        This is a simplified version for F = K.
        """
        FK_power = F**((1 - self.beta))
        
        correction = ((1 - self.beta)**2 / 24 * (self.alpha**2) / (FK_power**2) +
                      self.rho * self.beta * self.nu * self.alpha / (4 * FK_power) +
                      (2 - 3 * self.rho**2) * self.nu**2 / 24) * ttm
        
        return self.alpha / FK_power * (1 + correction)


class NormalSABRModel(SABRModel):
    """
    Normal SABR Model (β = 0).
    
    The forward price follows a normal (Bachelier) process:
    dF = α * dW1
    dα = ν * α * dW2
    """
    
    @property
    def beta(self) -> float:
        return 0.0
    
    def implied_volatility(self, F: float, K: float, ttm: float) -> float:
        """
        Compute implied volatility for Normal SABR.
        
        For β = 0, the formula simplifies.
        """
        if F <= 0 or K <= 0:
            raise ValueError("Forward and strike prices must be positive")
        if ttm <= 0:
            raise ValueError("Time to maturity must be positive")
        
        # Handle at-the-money case
        if abs(F - K) < 1e-10:
            return self._atm_volatility(F, ttm)
        
        # For normal SABR, z simplifies
        z = self.nu / self.alpha * (F - K)
        chi_z = self._compute_chi(z)
        
        if abs(chi_z) < 1e-10:
            return self._atm_volatility(F, ttm)
        
        # Main term (simplified for β = 0)
        main_term = self.alpha * z / chi_z
        
        # Correction terms
        correction = self._compute_correction_normal(F, K, ttm, z, chi_z)
        
        return main_term * (1 + correction)
    
    def _compute_correction_normal(self, F: float, K: float, ttm: float,
                                   z: float, chi_z: float) -> float:
        """Correction terms for Normal SABR (β = 0)."""
        # Simplified correction for β = 0
        term1 = (self.rho * self.nu * self.alpha / 4 +
                 (2 - 3 * self.rho**2) * self.nu**2 / 24) * ttm
        
        return term1
    
    def _atm_volatility(self, F: float, ttm: float) -> float:
        """ATM volatility for Normal SABR."""
        correction = (self.rho * self.nu * self.alpha / 4 +
                     (2 - 3 * self.rho**2) * self.nu**2 / 24) * ttm
        
        return self.alpha * (1 + correction)


class LognormalSABRModel(SABRModel):
    """
    Lognormal SABR Model (β = 1).
    
    The forward price follows a lognormal (Black-Scholes-like) process:
    dF = α * F * dW1
    dα = ν * α * dW2
    """
    
    @property
    def beta(self) -> float:
        return 1.0
        super().__init__(alpha=alpha, beta=1.0, rho=rho, nu=nu, r=r, F0=F0)
    
    def implied_volatility(self, F: float, K: float, ttm: float) -> float:
        """
        Compute implied volatility for Lognormal SABR.
        
        For β = 1, the formula simplifies and z = (ν/α) * ln(F/K).
        """
        if F <= 0 or K <= 0:
            raise ValueError("Forward and strike prices must be positive")
        if ttm <= 0:
            raise ValueError("Time to maturity must be positive")
        
        # Handle at-the-money case
        if abs(F - K) < 1e-10:
            return self._atm_volatility(F, ttm)
        
        # For lognormal SABR, z = (ν/α) * ln(F/K)
        z = self.nu / self.alpha * np.log(F / K)
        chi_z = self._compute_chi(z)
        
        if abs(chi_z) < 1e-10:
            return self._atm_volatility(F, ttm)
        
        # Main term (simplified for β = 1: (F*K)^0 = 1)
        main_term = self.alpha * z / chi_z
        
        # Correction terms
        correction = self._compute_correction_lognormal(F, K, ttm, z, chi_z)
        
        return main_term * (1 + correction)
    
    def _compute_correction_lognormal(self, F: float, K: float, ttm: float,
                                      z: float, chi_z: float) -> float:
        """Correction terms for Lognormal SABR (β = 1)."""
        log_FK = np.log(F / K)
        
        # For β = 1, the (1-β)^2 terms vanish
        term1 = (self.rho * self.nu * self.alpha / 4 +
                 (2 - 3 * self.rho**2) * self.nu**2 / 24) * ttm
        
        # Additional log terms
        if abs(z) > 1e-6:
            term2 = (log_FK**2) / 24 + (log_FK**4) / 1920
        else:
            term2 = 0
        
        return term1 + term2
    
    def _atm_volatility(self, F: float, ttm: float) -> float:
        """ATM volatility for Lognormal SABR."""
        correction = (self.rho * self.nu * self.alpha / 4 +
                     (2 - 3 * self.rho**2) * self.nu**2 / 24) * ttm
        
        return self.alpha * (1 + correction)
    
class ShiftedSABRModel(SABRModel):
    """
    Shifted SABR model: applies a constant shift to F and K
    before delegating pricing to an underlying SABR model.
    """

    def __init__(self, base_model: SABRModel, shift: float):
        if shift <= 0.0:
            raise ValueError("Shift must be positive")

        self.base_model = base_model
        self.shift = shift

        # initialize SABRModel with the same parameters
        super().__init__(
            alpha=base_model.alpha,
            rho=base_model.rho,
            nu=base_model.nu,
            r=base_model.r,
            F0=(
                base_model.F0 + shift
                if base_model.F0 is not None
                else None
            ),
        )

    @property
    def beta(self) -> float:
        return self.base_model.beta
    
    def implied_volatility(self, F: float, K: float, ttm: float) -> float:
        F_shifted = F + self.shift
        K_shifted = K + self.shift

        if F_shifted <= 0 or K_shifted <= 0:
            raise ValueError("Shifted F and K must be positive")

        return self.base_model.implied_volatility(
            F_shifted, K_shifted, ttm
        )
    
    def get_parameters(self):
        params = self.base_model.get_parameters().copy()
        params["shift"] = self.shift
        return params




