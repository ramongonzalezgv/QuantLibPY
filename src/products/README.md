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
- `__init__(S, K, expiry_date, valuation_date=None, calendar=None, option_type='call', qty=1, F=None)`: supports spot `S`, strike `K`, expiry date (string or `date` object), optional valuation date (defaults to today), optional market calendar, call/put, position size, and optional forward price `F`.
- `ttm`: time to maturity in years (computed from valuation_date and expiry_date).
- `valuation_date`: the date at which the option is valued (defaults to today).
- `expiry_date`: the option's expiration date.
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
- `__init__(S, K, expiry_date, valuation_date=None, calendar=None, option_type='call', qty=1, averaging_type='arithmetic')`: supports arithmetic or geometric averaging, along with expiry date, optional valuation date, and optional market calendar.
- `ttm`: years to maturity (computed from valuation_date and expiry_date).
- `get_parameters`: contract identifiers for caching.
- `payoff(spot_prices)`: expects an array shaped `(n_paths, n_steps + 1)` and applies arithmetic or geometric averaging per path before the vanilla-style payoff.
- `info()`: human-readable description.

## Usage examples
Simple European call payoff on terminal spots:
```python
import numpy as np
from datetime import date
from products.EuropeanOption import EuropeanOption

spots = np.array([90, 100, 105, 120])
valuation_date = date(2025, 1, 15)
expiry_date = date(2025, 2, 14)  # ~30 days later
option = EuropeanOption(S=100, K=100, expiry_date=expiry_date, valuation_date=valuation_date, option_type="call", qty=1)
payoffs = option.payoff(spots)  # array([0, 0, 5, 20])
```

Geometric-average Asian put over simulated paths:
```python
import numpy as np
from datetime import date
from products.AsianOption import AsianOption

# 3 paths, 5 observation points each
paths = np.array([
    [100, 98, 97, 96, 95],
    [100, 101, 103, 105, 104],
    [100, 99, 100, 99, 100],
])
valuation_date = date(2025, 1, 15)
expiry_date = date(2025, 4, 15)  # ~90 days later
asian_put = AsianOption(S=100, K=100, expiry_date=expiry_date, valuation_date=valuation_date, option_type="put", averaging_type="geometric")
payoffs = asian_put.payoff(paths)  # payoff per path after averaging
```

American option creation (exercise policy handled by engine):
```python
from datetime import date, timedelta
from products.AmericanOption import AmericanOption

valuation_date = date(2025, 1, 15)
expiry_date = valuation_date + timedelta(days=180)
am_call = AmericanOption(S=50, K=45, expiry_date=expiry_date, valuation_date=valuation_date, option_type="call")
# Pricing engine decides optimal early exercise; payoff definition matches European.
```