# Models: stochastic dynamics for pricing
The `models` package encodes the stochastic processes used by pricing engines. Each model implements a characteristic function and exposes its defining parameters for caching and reproducibility.

## Table of Contents
1. [StochasticModel (base class)](#stochasticmodel-base-class)
2. [BlackScholesModel](#blackscholesmodel)
3. [BlackModel](#blackmodel)
4. [HestonModel](#hestonmodel)
5. [SABRModel](#sabrmodel)
6. [Usage examples](#usage-examples)

## StochasticModel (base class)
Abstract API all models follow (`StochasticModel.py`):
```python
class StochasticModel(ABC):
    @abstractmethod
    def characteristic_function(self, u: complex, params: Dict) -> complex:
        """Model characteristic function evaluated at u."""

    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """Return model-defining inputs for cache keys."""
```
- `characteristic_function(u, params)`: returns ϕ(u; t) used by FFT/analytical engines. `params` typically includes `ttm` and, for Black, `F`.
- `get_parameters()`: intrinsic model inputs only (vols, rates, correlations) to keep cache keys stable.

## BlackScholesModel
Classic lognormal diffusion with dividends (`BlackScholesModel.py`).
- `__init__(sigma, r, q=0)`: spot volatility, risk-free rate, continuous dividend yield.
- `characteristic_function(u, params)`: returns ϕ(u) for time-to-maturity `params['ttm']`.
- `d1`/`d2(S, K, ttm)`: helpers for closed-form greeks/prices.

## BlackModel
Black (76) model for forwards/futures (`BlackModel.py`).
- `__init__(sigma, r)`: forward volatility and discount rate.
- `characteristic_function(u, params)`: expects `params['F']` and `params['ttm']`; used by FFT pricers.
- `d1`/`d2(F, K, ttm)`: Black closed-form helpers.

## HestonModel
Stochastic volatility with mean reversion (`HestonModel.py`).
- `__init__(kappa, theta, sigma, rho, v0, r, q=0)`: mean-reversion speed/level, vol-of-vol, correlation, initial variance, risk-free rate, dividend yield.
- `characteristic_function(u, params)`: semi-closed form ϕ(u) for `params['ttm']`; supports FFT/analytical engines.
- `get_parameters()`: all dynamics inputs for cache keys.

## SABRModel
SABR (Stochastic Alpha Beta Rho) model for forward prices (`SABRModel.py`).

The SABR model is a stochastic volatility model where both the forward price and its volatility follow stochastic processes:
- `dF = α * F^β * dW1`
- `dα = ν * α * dW2`
- `dW1 * dW2 = ρ * dt`

**Base class:** `SABRModel`
- `__init__(alpha, beta, rho, nu, r, F0=None)`: initial volatility, CEV exponent (0≤β≤1), correlation, vol-of-vol, risk-free rate, optional initial forward.
- `implied_volatility(F, K, ttm)`: computes Black implied volatility using Hagan's 2002 approximation.
- `get_parameters()`: returns model parameters for caching.
- **Note:** SABR does not have a closed-form characteristic function. Pricing is done via `AnalyticalEngine`, which uses SABR's implied volatility with Black's formula (same approach as `BlackModel`).

**Specialized classes:**
- `NormalSABRModel(alpha, rho, nu, r, F0=None)`: Normal SABR (β=0), forward follows normal process.
- `LognormalSABRModel(alpha, rho, nu, r, F0=None)`: Lognormal SABR (β=1), forward follows lognormal process.

**Reference:** Hagan, P. S., Kumar, D., Lesniewski, A. S., & Woodward, D. E. (2002). "Managing smile risk." The Best of Wilmott, 1, 249-296.

**Note:** SABR models work with forward prices. Set `product.F` when using with `EuropeanOption`, or the spot price will be used as the forward.

## Usage examples
Evaluate characteristic functions (typical entry point for FFT engines):
```python
from models.BlackScholesModel import BlackScholesModel
from models.BlackModel import BlackModel
from models.HestonModel import HestonModel

bs = BlackScholesModel(sigma=0.2, r=0.01, q=0.0)
phi_bs = bs.characteristic_function(u=1j, params={"ttm": 0.5})

black = BlackModel(sigma=0.25, r=0.015)
phi_black = black.characteristic_function(u=1j, params={"ttm": 1.0, "F": 102})

heston = HestonModel(kappa=2.0, theta=0.04, sigma=0.6, rho=-0.7, v0=0.04, r=0.01, q=0.0)
phi_heston = heston.characteristic_function(u=1j, params={"ttm": 0.75})
```

Use `d1`/`d2` helpers for closed-form Black-Scholes/Black pricing:
```python
S, K, ttm = 100, 105, 0.5
d1 = bs.d1(S, K, ttm)
d2 = bs.d2(S, K, ttm)

F = 102
bd1 = black.d1(F, K=100, ttm=1.0)
bd2 = black.d2(F, K=100, ttm=1.0)
```

SABR pricing using AnalyticalEngine (same pattern as other models):
```python
from datetime import date, timedelta
from models.SABRModel import LognormalSABRModel, NormalSABRModel
from products.EuropeanOption import EuropeanOption
from engines.AnalyticalEngine import AnalyticalEngine

valuation_date = date(2025, 1, 15)
expiry_date = valuation_date + timedelta(days=91)  # ~0.25 years

# Lognormal SABR (β=1)
sabr_ln = LognormalSABRModel(alpha=0.2, rho=-0.5, nu=0.3, r=0.05, F0=100)
opt = EuropeanOption(S=100, K=100, expiry_date=expiry_date, valuation_date=valuation_date, option_type='call', F=100)
engine = AnalyticalEngine()
price_ln = engine.calculate_price(opt, sabr_ln)

# Normal SABR (β=0)
sabr_norm = NormalSABRModel(alpha=0.2, rho=-0.4, nu=0.25, r=0.05, F0=100)
price_norm = engine.calculate_price(opt, sabr_norm)

# Direct implied volatility computation (for analysis/calibration)
implied_vol = sabr_ln.implied_volatility(F=100, K=105, ttm=0.25)
print(f"Implied volatility: {implied_vol:.4f}")
```

**Architecture Note:** Like `BlackModel`, SABR models provide implied volatility, and `AnalyticalEngine` applies Black's formula. The model does not contain pricing logic—this separation maintains consistency with other models in the framework.
