# Products: core abstractions and concrete options
The `products` package defines the contract objects that pricing engines consume. Each product implements `payoff` (how cash flows depend on simulated prices) and `get_parameters` (a stable description used for caching and reproducibility).

## Table of Contents
1. [FinancialProduct (base class)](#financialproduct-base-class)
2. [EuropeanOption](#europeanoption)
3. [AmericanOption](#americanoption)
4. [AsianOption](#asianoption)
5. [Usage examples](#usage-examples)

## FinancialProduct (base class)
Abstract interface every product must follow (`FinancialProduct.py`):
```python
class FinancialProduct(ABC):
    @abstractmethod
    def payoff(self, spot_prices: np.ndarray) -> np.ndarray:
        """Return payoff per simulated path."""

    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """Return contract-defining params for cache keys."""
```
- `payoff`: takes simulated spot paths and returns one payoff per path.
- `get_parameters`: returns only intrinsic contract inputs (e.g., strike, maturity, type) so pricing caches remain stable across runs.

## EuropeanOption
Vanilla European call/put (`EuropeanOption.py`).
- `__init__(S, K, T, option_type='call', qty=1, premium=None, F=None)`: supports spot `S`, strike `K`, maturity `T` (days or `dd-mm-YYYY`), call/put, position size, optional premium, and optional forward price `F`.
- `ttm`: time to maturity in years (computed from `T`).
- `maturity_date`: expiry as a `date` when `T` is a day-count.
- `F`: forward price when provided.
- `get_parameters`: stable identifier for caching.
- `payoff(spot_prices)`: `max(S_T - K, 0)` for calls or `max(K - S_T, 0)` for puts, scaled by `qty`.
- `info()`: formatted string summary.

## AmericanOption
American-style option (`AmericanOption.py`).
- Inherits all arguments and behavior from `EuropeanOption`.
- `info()` only changes the label to “American”. Early-exercise logic is handled by the selected pricing engine, not the product.

## AsianOption
Average-price Asian option (`AsianOption.py`).
- `__init__(S, K, T, option_type='call', qty=1, averaging_type='arithmetic')`: supports arithmetic or geometric averaging.
- `ttm`: years to maturity (same logic as European).
- `get_parameters`: contract identifiers for caching.
- `payoff(spot_prices)`: expects an array shaped `(n_paths, n_steps + 1)` and applies arithmetic or geometric averaging per path before the vanilla-style payoff.
- `info()`: human-readable description.

## Usage examples
Simple European call payoff on terminal spots:
```python
import numpy as np
from products.EuropeanOption import EuropeanOption

spots = np.array([90, 100, 105, 120])
option = EuropeanOption(S=100, K=100, T=30, option_type="call", qty=1)
payoffs = option.payoff(spots)  # array([0, 0, 5, 20])
```

Geometric-average Asian put over simulated paths:
```python
import numpy as np
from products.AsianOption import AsianOption

# 3 paths, 5 observation points each
paths = np.array([
    [100, 98, 97, 96, 95],
    [100, 101, 103, 105, 104],
    [100, 99, 100, 99, 100],
])
asian_put = AsianOption(S=100, K=100, T=90, option_type="put", averaging_type="geometric")
payoffs = asian_put.payoff(paths)  # payoff per path after averaging
```

American option creation (exercise policy handled by engine):
```python
from products.AmericanOption import AmericanOption

am_call = AmericanOption(S=50, K=45, T=180, option_type="call")
# Pricing engine decides optimal early exercise; payoff definition matches European.
```