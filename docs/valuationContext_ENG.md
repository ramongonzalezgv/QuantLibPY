# OptionValuationContext: Valuation Orchestration with Caching and Parallelization

## Table of Contents
1. [Introduction](#introduction)
2. [Architecture](#architecture)
3. [How Caching Works](#how-caching-works)
4. [How OptionValuationContext Works](#how-optionvaluationcontext-works)
5. [Usage Examples](#usage-examples)
6. [Advantages and Use Cases](#advantages-and-use-cases)

---

## Introduction

**OptionValuationContext** is a class that orchestrates option valuation (pricing) by combining three main components:

- **Engine**: the calculation engine (Analytical, FFT, Monte Carlo)
- **Model**: the stochastic model (Black-Scholes, Heston, etc.)
- **Product**: the product (EuropeanOption, etc.)

It also provides:
- **Thread-safe LRU Cache**: avoids recalculating the same product+model combinations
- **Integrated Logging**: for debugging and auditing
- **Parallel Execution**: valuation of multiple products in parallel (threads or processes)
- **Uniform API**: `value_option()` for a single product, `value_options()` for batches

---

## Architecture

### Key Components

#### 1. **_make_cache_key(product, model, extra_kwargs)**

Generates a unique key (SHA256) from the product, model, and extra parameters.

```python
def _make_cache_key(product: Any, model: Any, extra_kwargs: Dict) -> str:
    """
    Extracts the representation of the product and model:
    - If product.get_parameters() exists, use it
    - Otherwise, use vars(product) (attributes)
    - Fallback to repr() if everything fails
    
    Serializes to JSON, hashes to SHA256 for a stable key.
    """
```

**Advantage**: two calls with exactly the same parameters will generate the same key.

#### 2. **_LRUCache**

Thread-safe LRU (Least Recently Used) cache based on `OrderedDict`.

```python
class _LRUCache:
    def __init__(self, maxsize: int = 1024):
        self.maxsize = maxsize        # maximum number of entries
        self._d = OrderedDict()       # ordered dictionary
        self._lock = threading.Lock() # concurrency protection

    def get(self, key: str):
        # Searches for the key; if found, moves it to the end (most recent)
        # Returns None if not found

    def set(self, key: str, value: Any):
        # Adds/updates key
        # If maxsize is exceeded, deletes the oldest (FIFO)
```

**LRU Algorithm**:
1. Maintains an ordered dictionary (FIFO → most recent at end)
2. When a key is accessed, it moves to the end (marks as recent)
3. When full, removes the first entry (oldest)

**Thread-safety**: uses `threading.Lock()` to protect concurrent reads/writes.

#### 3. **OptionValuationContext**

Main orchestrator that:
- Stores the engine, logger, cache, and parallelization config
- Exposes `value_option()` (single) and `value_options()` (batch)
- Manages cache hits/misses
- Executes in parallel if requested

---

## How Caching Works

### Valuation Flow with Cache

```
┌─────────────────────────────────────────┐
│ value_option(product, model, **kwargs)  │
└────────────────┬────────────────────────┘
                 │
                 ▼
        ┌────────────────────┐
        │ Cache enabled?     │
        └────┬───────────┬───┘
           NO│           │YES
             │           ▼
             │    ┌──────────────────────┐
             │    │ Generate SHA256 key  │
             │    │ (product+model+kw)   │
             │    └──────────┬───────────┘
             │               │
             │               ▼
             │        ┌──────────────────┐
             │        │ Search in cache  │
             │        └────┬──────────┬──┘
             │          HIT│          │MISS
             │             │          │
             │             ▼          ▼
             │        ┌─────────┐  ┌─────────────────────┐
             │        │ Return  │  │ Calculate with eng. │
             │        │ value   │  │ (calculate_price)   │
             │        └─────────┘  └──────────┬──────────┘
             │                                 │
             └─────────────────┬───────────────┘
                               │
                               ▼
                      ┌────────────────────┐
                      │ Save to cache?     │
                      │ (if key valid)     │
                      └────┬────────┬──────┘
                         YES│        │ NO
                           ▼        │
                      ┌─────────┐   │
                      │cache.set│   │
                      │(k, val) │   │
                      └─────────┘   │
                                    │
                      ┌─────────────┘
                      │
                      ▼
                  ┌─────────┐
                  │ Return  │
                  │ price   │
                  └─────────┘
```

### Cache Hit/Miss Example

```python
# First call → MISS (calculates)
option = EuropeanOption(S=100, K=100, T=30, option_type='call', qty=1)
bs_model = BlackScholesModel(sigma=0.2, r=0.05, q=0.02)

ctx = OptionValuationContext(analytical_engine, cache_enabled=True)

p1 = ctx.value_option(option, bs_model)  # → MISS
# Generates SHA256 key from (option, bs_model)
# Not in cache → calls engine.calculate_price()
# Saves result to cache

p2 = ctx.value_option(option, bs_model)  # → HIT
# Generates same key (same product/model)
# Found in cache → returns directly without calculating
# p1 == p2 (exactly the same value)

# Third call with different K → MISS
option2 = EuropeanOption(S=100, K=110, T=30, option_type='call', qty=1)
p3 = ctx.value_option(option2, bs_model)  # → MISS
# Generates DIFFERENT key (K=110 vs K=100)
# Not in cache → calculates again
# p3 ≠ p1 (different prices)
```

### Cache Limitations

1. **Immutable Parameters**: if you modify `product` or `model` after creating the context, the cache won't automatically invalidate.
   ```python
   opt = EuropeanOption(S=100, K=100, T=30, option_type='call', qty=1)
   p1 = ctx.value_option(opt, bs_model)  # HIT/MISS depends on history
   
   opt.S = 110  # MANUAL MODIFICATION (not recommended)
   p2 = ctx.value_option(opt, bs_model)  # May be in cache with S=100
   ```
   **Solution**: create new objects instead of modifying.

2. **Limited Size**: when `cache_maxsize` is reached, the oldest entry is deleted.
   ```python
   ctx = OptionValuationContext(..., cache_maxsize=128)
   # If you do 129 distinct valuations, the first one is removed from cache
   ```

3. **No Manual Invalidation**: there's no explicit cache clearing method (could be added if needed).

---

## How OptionValuationContext Works

### value_option(): Single Valuation

```python
def value_option(
    self,
    product: Any,
    model: Any,
    *,
    use_cache: Optional[bool] = None,
    log: bool = True,
    **kwargs
) -> float:
```

**Parameters**:
- `product`: EuropeanOption, etc.
- `model`: BlackScholesModel, HestonModel, etc.
- `use_cache`: override cache_enabled for this call (None = use default)
- `log`: if False, doesn't log to logger (default True)
- `**kwargs`: extra parameters for engine.calculate_price()

**Logic**:
1. Determines whether to use cache (use_cache if specified, otherwise cache_enabled)
2. If cache enabled:
   - Generates SHA256 key
   - Attempts to get value from cache
   - If hit, returns (without calculating)
3. Logs to logger (debug): what will be valued
4. Calls `engine.calculate_price(product, model, **kwargs)`
5. If cache enabled and key valid, saves result
6. Logs computed price
7. Returns price

### value_options(): Batch Valuation

```python
def value_options(
    self,
    products: Iterable[Any],
    model: Any,
    *,
    use_cache: Optional[bool] = None,
    parallel: Optional[bool] = None,
    progress_callback: Optional[Callable[[int, float], None]] = None,
    **kwargs
) -> List[float]:
```

**Parameters**:
- `products`: list/iterable of products
- `model`: single model (same for all)
- `use_cache`: override cache_enabled
- `parallel`: override self.parallel (True/False)
- `progress_callback`: function `f(idx, price)` called when each price is ready
- `**kwargs`: parameters for engine

**Sequential Flow** (parallel=False):
```
for i, product in enumerate(products):
    price = value_option(product, model, ...)
    results.append(price)
    if progress_callback:
        progress_callback(i, price)
return results
```

**Parallel Flow** (parallel=True):
```
Create ThreadPoolExecutor (or ProcessPoolExecutor if use_process_pool=True)
├─ Submit task for each (idx, product)
├─ Task = value_option(product, model, ...)
├─ Wait for completion (as_completed)
├─ Store result in original order (results[idx])
└─ Call progress_callback when each task completes
return results (in same order as input)
```

### Parallelization Benefits

| Case | Gain |
|------|------|
| **Analytical Engine** | Minimal (fast calculation, overhead > benefit) |
| **Monte Carlo** | **High** (long simulations, CPU-bound) |
| **FFT** | **Medium** (fast transforms but multiple strikes) |

**Example**: valuing 100 options with Monte Carlo:
- Without parallelization: ~100 * (time per option)
- With 4 workers: ~25 * (time per option) + overhead = **~4x faster**

### Engine Switching

A key advantage of using OptionValuationContext is that you can switch engines at runtime **without changing client code**:

```python
# Use the same context, switch engine
ctx = OptionValuationContext(AnalyticalEngine())
p1 = ctx.value_option(opt, bs_model)  # Analytical

ctx.engine = FFTEngine()
p2 = ctx.value_option(opt, bs_model)  # FFT (same context)

ctx.engine = MonteCarloEngine(n_paths=100000, seed=42)
p3 = ctx.value_option(opt, bs_model)  # MC (same context)

# Compare without changing client code
print(f"Diff Analytical vs FFT: {abs(p1 - p2):.6e}")
print(f"Diff Analytical vs MC: {abs(p1 - p3):.6e}")
```

---

## Usage Examples

### Example 1: Simple Caching

```python
from src.valuation.context import OptionValuationContext
from src.engines.engines import AnalyticalEngine
from src.products.products import EuropeanOption
from src.models.models import BlackScholesModel

engine = AnalyticalEngine()
ctx = OptionValuationContext(engine, cache_enabled=True, cache_maxsize=128)

opt = EuropeanOption(S=100, K=100, T=30, option_type='call', qty=1)
bs = BlackScholesModel(sigma=0.2, r=0.05, q=0.02)

# First call: MISS (calculates)
p1 = ctx.value_option(opt, bs)
print(f"Price 1: {p1:.6f}")

# Second call: HIT (from cache, instant)
p2 = ctx.value_option(opt, bs)
print(f"Price 2: {p2:.6f}")

assert p1 == p2  # exactly equal
```

### Example 2: Sequential Batch

```python
import numpy as np

strikes = np.linspace(80, 120, 10)
products = [
    EuropeanOption(S=100, K=K, T=30, option_type='call', qty=1)
    for K in strikes
]

ctx = OptionValuationContext(engine, cache_enabled=False, parallel=False)
prices = ctx.value_options(products, bs)

for K, p in zip(strikes, prices):
    print(f"K={K:.1f}: {p:.6f}")
```

### Example 3: Parallel Batch with Progress Callback

```python
def on_progress(idx, price):
    print(f"[✓] Option {idx} completed: {price:.6f}")

ctx = OptionValuationContext(
    MonteCarloEngine(n_paths=50000, seed=42),
    parallel=True,
    max_workers=4
)

prices = ctx.value_options(
    products,
    bs,
    progress_callback=on_progress
)
```

### Example 4: Engine Switching

```python
ctx = OptionValuationContext(AnalyticalEngine())

# Compare three engines on the same product
engines_to_test = [
    ('Analytical', AnalyticalEngine()),
    ('FFT', FFTEngine()),
    ('Monte Carlo', MonteCarloEngine(n_paths=100000, seed=42)),
]

results = {}
for name, eng in engines_to_test:
    ctx.engine = eng
    price = ctx.value_option(opt, bs)
    results[name] = price
    print(f"{name}: {price:.6f}")

# Analyze differences
analytical = results['Analytical']
for name, price in results.items():
    if name != 'Analytical':
        diff = abs(analytical - price)
        rel_diff = 100 * diff / analytical
        print(f"  {name} vs Analytical: {diff:.6e} ({rel_diff:.2f}%)")
```

### Example 5: Logging for Debugging

```python
import logging

# Configure logging at DEBUG level to see cache hits/misses
logging.basicConfig(level=logging.DEBUG)

ctx = OptionValuationContext(
    engine,
    cache_enabled=True,
    cache_maxsize=10
)

# Each call will be logged
p1 = ctx.value_option(opt1, bs)  # DEBUG: Cache key generation, valuation...
p2 = ctx.value_option(opt1, bs)  # DEBUG: Cache hit for key=...
p3 = ctx.value_option(opt2, bs)  # DEBUG: Cache miss, valuation...
```

---

## Advantages and Use Cases

### ✅ Main Advantages

| Advantage | Description |
|-----------|-------------|
| **Decoupling** | Separates orchestration (context) from pricing (engine) |
| **Reusability** | The same context works with any engine |
| **Caching** | Avoids recalculating identical products |
| **Logging** | Centralized auditing and debugging |
| **Parallelization** | Batch valuations accelerated (especially MC) |
| **Flexibility** | Switch engines without modifying client code |
| **Uniformity** | Single API (value_option / value_options) |

### 📊 Use Cases

#### 1. **Backtesting and Sensitivity Analysis**
```python
# Value many options with different strikes/maturities
strikes = np.linspace(80, 120, 50)
maturities = [7, 30, 90, 180, 365]

all_products = [
    EuropeanOption(S=100, K=K, T=T, option_type='call', qty=1)
    for K in strikes
    for T in maturities
]

ctx = OptionValuationContext(
    MonteCarloEngine(n_paths=10000, seed=42),
    parallel=True,
    max_workers=8,
    cache_enabled=True
)

prices = ctx.value_options(all_products, heston_model)
# → 250 valuations, parallelized and cached
```

#### 2. **Model Comparison**
```python
models_to_compare = {
    'BS': BlackScholesModel(sigma=0.2, r=0.05, q=0.02),
    'Heston': HestonModel(kappa=2.0, theta=0.04, ...),
}

ctx = OptionValuationContext(FFTEngine())

results = {}
for model_name, model in models_to_compare.items():
    prices = ctx.value_options(products, model)
    results[model_name] = prices
```

#### 3. **Parameter Calibration**
```python
from scipy.optimize import minimize

def objective(params):
    # Create model with candidate parameters
    model = HestonModel(*params, r=0.05, q=0.02)
    
    # Value with context
    ctx.engine = MonteCarloEngine(n_paths=5000, seed=42)
    model_prices = ctx.value_options(market_products, model)
    
    # Minimize difference with market prices
    error = np.sum((np.array(model_prices) - np.array(market_prices))**2)
    return error

result = minimize(objective, x0=[1.0, 0.04, 0.3, -0.5, 0.04])
```

#### 4. **Risk Management (Greeks)**
```python
# Calculate greeks for a portfolio of options
portfolio_options = [...]

ctx = OptionValuationContext(AnalyticalEngine(), cache_enabled=True)

deltas = [GreeksCalculator.delta(opt, model) for opt in portfolio_options]
gammas = [GreeksCalculator.gamma(opt, model) for opt in portfolio_options]
vegas = [GreeksCalculator.vega(opt, model) for opt in portfolio_options]

# Cache accelerates these calculations (many numerical derivatives)
```

### 🎯 When to Use the Context

**Use OptionValuationContext when**:
- ✅ You need to value multiple products (batch)
- ✅ You want to compare engines or models
- ✅ You require parallelization
- ✅ You need caching to optimize performance
- ✅ You want centralized auditing/logging

**Use engine directly when**:
- ✅ A single simple valuation
- ✅ Engine is already simple (overhead not worth it)
- ✅ You don't need cache or parallelization

---

## Conclusion

OptionValuationContext is an **orchestration and optimization** tool that:
1. **Simplifies** the API (value_option / value_options)
2. **Optimizes** performance (caching + parallelization)
3. **Facilitates** testing and comparison (engine switching)
4. **Centralizes** logging and auditing
5. **Scales** from simple valuations to complex analysis

Its modular architecture allows your project to grow without refactoring client code.