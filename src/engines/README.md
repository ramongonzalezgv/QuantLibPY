# Engines: pricing methods
The `engines` package implements pricing algorithms (analytical, tree-based, FFT, Monte Carlo). Every engine follows `PricingEngine.calculate_price(product, model) -> float`.

## Table of Contents
1. [PricingEngine (base class)](#pricingengine-base-class)
2. [AnalyticalEngine](#analyticalengine)
3. [BinomialEngine](#binomialengine)
4. [FFTEngine](#fftengine)
5. [MonteCarloEngine](#montecarloengine)
6. [Usage examples](#usage-examples)

## PricingEngine (base class)
Abstract interface (`PricingEngine.py`):
```python
class PricingEngine(ABC):
    @abstractmethod
    def calculate_price(self, product, model) -> float:
        """Return the product price under the given model."""
```

## AnalyticalEngine
Closed-form pricing for Black-Scholes and Black (76) (`AnalyticalEngine.py`).
- Accepts `BlackScholesModel` or `BlackModel`; raises otherwise.
- Uses product fields (`S`, `K`, `F`, `ttm`, `option_type`, `qty`) and model params (`r`, `q`, `sigma`) to compute vanilla call/put prices.
- Relies on model `d1`/`d2` helpers.

## BinomialEngine
Cox-Ross-Rubinstein tree, suitable for European and American-style payoffs (`BinomialEngine.py`).
- `n_steps`: tree depth (accuracy vs speed).
- Uses model rates/vol (`r`, `q`, `sigma`) and product payoff to backward-induce prices.
- `price_and_levels` caches computations and can return early-level node values and up/down factors.

## FFTEngine
Fourier pricing via Lewis’ method (`FFTEngine.py`).
- Works with `BlackScholesModel`, `BlackModel`, and `HestonModel` using their characteristic functions.
- Key params: grid size `N`, range `B`, interpolation `interp`.
- Computes call prices; converts to put via parity.

## MonteCarloEngine
Path simulation pricing (`MonteCarloEngine.py`).
- Supports `EuropeanOption` and `AsianOption` with `BlackScholesModel` or `HestonModel`.
- Params: `n_paths`, `n_steps`, optional `seed`.
- Uses exact or stepwise schemes for BS; Euler with truncation for Heston.
- Discounts the average payoff at risk-free rate.

## Tuning tips
- FFT (`FFTEngine`): increase `N` (power of two) for resolution; widen `B` to capture far wings. Watch log-moneyness interpolation—very deep ITM/OTM strikes may need larger `B` or a different `interp` (e.g., `linear` for stability).
- Monte Carlo: raise `n_paths` to reduce Monte Carlo error; raise `n_steps` for path-dependent payoffs (Asians). Use `seed` to reproduce results; antithetic/control variates are not built-in—add them if you need faster convergence.
- Binomial: `n_steps` controls accuracy; American options typically need more steps than European. Ensure `n_steps` keeps `dt` small enough for the target maturity.

## Usage examples
Analytical Black-Scholes pricing for a European call:
```python
from datetime import date, timedelta
from products.EuropeanOption import EuropeanOption
from models.BlackScholesModel import BlackScholesModel
from engines.AnalyticalEngine import AnalyticalEngine

valuation_date = date(2025, 1, 15)
expiry_date = valuation_date + timedelta(days=30)
option = EuropeanOption(S=100, K=100, expiry_date=expiry_date, valuation_date=valuation_date, option_type="call", qty=1)
model = BlackScholesModel(sigma=0.2, r=0.01, q=0.0)
engine = AnalyticalEngine()
price = engine.calculate_price(option, model)
```

FFT pricing with Heston:
```python
from datetime import date, timedelta
from products.EuropeanOption import EuropeanOption
from models.HestonModel import HestonModel
from engines.FFTEngine import FFTEngine

valuation_date = date(2025, 1, 15)
expiry_date = valuation_date + timedelta(days=180)
option = EuropeanOption(S=100, K=95, expiry_date=expiry_date, valuation_date=valuation_date, option_type="call", qty=1)
model = HestonModel(kappa=2.0, theta=0.04, sigma=0.6, rho=-0.7, v0=0.04, r=0.01, q=0.0)
engine = FFTEngine(N=2**12, B=200)
price = engine.calculate_price(option, model)
```

Monte Carlo pricing for an arithmetic Asian call under BS:
```python
import numpy as np
from datetime import date, timedelta
from products.AsianOption import AsianOption
from models.BlackScholesModel import BlackScholesModel
from engines.MonteCarloEngine import MonteCarloEngine

valuation_date = date(2025, 1, 15)
expiry_date = valuation_date + timedelta(days=365)
option = AsianOption(S=100, K=100, expiry_date=expiry_date, valuation_date=valuation_date, option_type="call", averaging_type="arithmetic")
model = BlackScholesModel(sigma=0.25, r=0.015, q=0.0)
engine = MonteCarloEngine(n_paths=5000, n_steps=252, seed=42)
price = engine.calculate_price(option, model)
```

Binomial example (American put):
```python
from datetime import date, timedelta
from products.AmericanOption import AmericanOption
from models.BlackScholesModel import BlackScholesModel
from engines.BinomialEngine import BinomialEngine

valuation_date = date(2025, 1, 15)
expiry_date = valuation_date + timedelta(days=180)
option = AmericanOption(S=50, K=55, expiry_date=expiry_date, valuation_date=valuation_date, option_type="put")
model = BlackScholesModel(sigma=0.3, r=0.02, q=0.0)
engine = BinomialEngine(n_steps=500)
price = engine.calculate_price(option, model)
```
