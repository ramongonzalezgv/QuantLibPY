# Models: stochastic dynamics for pricing
The `models` package encodes the stochastic processes used by pricing engines. Each model implements a characteristic function and exposes its defining parameters for caching and reproducibility.

## Table of Contents
1. [StochasticModel (base class)](#stochasticmodel-base-class)
2. [BlackScholesModel](#blackscholesmodel)
3. [BlackModel](#blackmodel)
4. [HestonModel](#hestonmodel)
5. [Usage examples](#usage-examples)

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
