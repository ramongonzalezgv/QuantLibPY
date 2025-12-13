# Greeks: sensitivity analysis for options

The `greeks` package computes option sensitivities (Greeks) that measure how option prices change with respect to underlying parameters. These metrics are essential for risk management, hedging, and portfolio analysis.

## Table of Contents
1. [Overview](#overview)
2. [What are Greeks?](#what-are-greeks)
3. [Architecture](#architecture)
4. [GreeksCalculator](#greekscalculator)
5. [Strategies](#strategies)
6. [Usage Examples](#usage-examples)
7. [Parameter Tuning](#parameter-tuning)

## Overview

The module provides a strategy-based architecture for computing Greeks:
- **GreeksCalculator**: main interface that selects and executes strategies
- **GreeksStrategy**: abstract base class for calculation methods
- **AnalyticalGreeksStrategy**: closed-form formulas (Black-Scholes, Black 76)
- **BinomialGreeksStrategy**: numerical methods using finite differences and tree extraction

Key features:
- Automatic strategy selection based on product/model compatibility
- Optional caching to avoid redundant calculations
- Richardson extrapolation for improved accuracy in binomial methods
- Support for European, American, and other option types

## What are Greeks?

Greeks measure the sensitivity of option prices to changes in market parameters:

| Greek | Symbol | Definition | Interpretation |
|-------|--------|------------|----------------|
| **Delta** | Δ | ∂Price/∂S | Change in option price per $1 move in underlying |
| **Gamma** | Γ | ∂²Price/∂S² | Rate of change of Delta (convexity) |
| **Theta** | Θ | -∂Price/∂t | Time decay (price change per year as time passes) |
| **Vega** | ν | ∂Price/∂σ | Sensitivity to volatility (per 1.0 vol change) |
| **Rho** | ρ | ∂Price/∂r | Sensitivity to risk-free rate (per 1.0 rate change) |

**Notes:**
- Delta ranges from 0 to 1 for calls, -1 to 0 for puts
- Gamma is always positive (same for calls and puts)
- Theta is typically negative (options lose value over time)
- Vega is always positive (higher vol → higher option value)
- Rho is positive for calls, negative for puts

## Architecture

### GreeksStrategy (base class)

Abstract interface that all strategies implement:

```python
class GreeksStrategy(ABC):
    @abstractmethod
    def calculate_greeks(self, product, model) -> Dict[str, float]:
        """Return dict with keys: Price, Delta, Gamma, Theta, Vega, Rho"""
```

### GreeksCalculator

Central manager that:
- Selects appropriate strategy based on product/model
- Caches results to avoid recomputation
- Provides a unified API (`all_greeks()`)

**Strategy Selection Logic:**
1. If `force_numerical=True` → always use binomial
2. If product is `AmericanOption` → use binomial (no analytical solution)
3. If `prefer_analytical=True` and product is `EuropeanOption` with `BlackScholesModel` → use analytical
4. Otherwise → fallback to binomial

## GreeksCalculator

Main entry point for computing Greeks.

### Initialization

```python
calculator = GreeksCalculator(
    default_n_steps=200,           # Binomial tree steps (if using binomial strategy)
    default_use_richardson=True,   # Use Richardson extrapolation for binomial
    enable_cache=True               # Enable result caching
)
```

### Methods

#### `all_greeks(product, model, **options) -> Dict[str, float]`

Computes all Greeks using the selected strategy.

**Parameters:**
- `product`: Option product (EuropeanOption, AmericanOption, etc.)
- `model`: Stochastic model (BlackScholesModel, HestonModel, etc.)
- `prefer_analytical`: Prefer closed-form when available (default: True)
- `force_numerical`: Force binomial strategy (default: False)
- `strategy_name`: Explicit strategy override ('analytical' | 'binomial' | custom)
- `n_steps`: Override binomial steps for this call
- `use_richardson`: Override Richardson usage for binomial
- `use_cache`: Override calculator cache setting

**Returns:** Dictionary with keys: `Price`, `Delta`, `Gamma`, `Theta`, `Vega`, `Rho`

#### `clear_cache()`

Clears the internal result cache.

#### `register_strategy(name, constructor)`

Register a custom Greeks calculation strategy.

## Strategies

### AnalyticalGreeksStrategy

**Use when:** European options with Black-Scholes or Black (76) models.

**Advantages:**
- Exact closed-form formulas (no numerical error)
- Very fast computation
- No discretization parameters to tune

**Limitations:**
- Only works with Black-Scholes and Black models
- Cannot handle American options (early exercise)
- Not applicable to stochastic volatility models (Heston)

**Implementation:**
- Uses model's `d1`/`d2` helpers
- Applies standard Black-Scholes Greek formulas
- Handles dividends (`q`) and forward prices (`F`)

**Formulas (Black-Scholes):**
- **Delta (call)**: `e^(-q*t) * N(d1)`
- **Delta (put)**: `e^(-q*t) * (N(d1) - 1)`
- **Gamma**: `e^(-q*t) * φ(d1) / (S * σ * √t)`
- **Vega**: `S * e^(-q*t) * φ(d1) * √t`
- **Theta**: Time decay term (varies by call/put)
- **Rho**: `K * t * e^(-r*t) * N(d2)` for calls, negative for puts

Where `N()` is CDF, `φ()` is PDF of standard normal.

### BinomialGreeksStrategy

**Use when:** American options, or when analytical formulas aren't available.

**Advantages:**
- Works with any model supported by BinomialEngine
- Handles early exercise (American options)
- Can extract Delta/Gamma directly from tree nodes (more accurate than finite differences)

**Limitations:**
- Slower than analytical (requires tree construction)
- Accuracy depends on `n_steps` parameter
- Finite difference approximations for Theta, Vega, Rho

**Implementation:**
1. **Price/Delta/Gamma**: Extracted from tree nodes (levels 0, 1, 2) when available
2. **Richardson Extrapolation**: Computes at N and 2N steps, extrapolates: `2*Price(2N) - Price(N)`
3. **Finite Differences**: Fallback for Delta/Gamma if tree extraction fails; always used for Theta, Vega, Rho

**Tree Extraction Method:**
- Builds CRR tree and extracts option values at root, level 1, and level 2
- Delta from level 1: `(V_up - V_down) / (S_up - S_down)`
- Gamma from level 2: second derivative approximation using three nodes

**Finite Difference Bumps:**
- **Delta**: `(Price(S+ε) - Price(S-ε)) / (2*ε)` where `ε = 0.1% * S`
- **Gamma**: `(Price(S+ε) - 2*Price(S) + Price(S-ε)) / ε²`
- **Theta**: `(Price(T-1day) - Price(T)) / (-1day/365)` (per year)
- **Vega**: `(Price(σ+0.01) - Price(σ-0.01)) / (2*0.01)`
- **Rho**: `(Price(r+1e-4) - Price(r-1e-4)) / (2*1e-4)`

## Usage Examples

### Example 1: Simple European Call (Analytical)

```python
from src.products.EuropeanOption import EuropeanOption
from src.models.BlackScholesModel import BlackScholesModel
from src.greeks.GreeksCalculator import GreeksCalculator

# Create option and model
option = EuropeanOption(S=100, K=100, T=30, option_type="call", qty=1)
model = BlackScholesModel(sigma=0.2, r=0.05, q=0.02)

# Calculate Greeks
calculator = GreeksCalculator()
greeks = calculator.all_greeks(option, model)

print(f"Price: {greeks['Price']:.4f}")
print(f"Delta: {greeks['Delta']:.4f}")
print(f"Gamma: {greeks['Gamma']:.6f}")
print(f"Theta: {greeks['Theta']:.4f}")
print(f"Vega: {greeks['Vega']:.4f}")
print(f"Rho: {greeks['Rho']:.4f}")
```

**Output:**
```
Price: 2.3456
Delta: 0.5234
Gamma: 0.012345
Theta: -0.0234
Vega: 0.1234
Rho: 0.0456
```

### Example 2: American Put (Binomial)

```python
from src.products.AmericanOption import AmericanOption
from src.models.BlackScholesModel import BlackScholesModel
from src.greeks.GreeksCalculator import GreeksCalculator

option = AmericanOption(S=50, K=55, T=180, option_type="put")
model = BlackScholesModel(sigma=0.3, r=0.02, q=0.0)

calculator = GreeksCalculator(default_n_steps=500, default_use_richardson=True)
greeks = calculator.all_greeks(option, model, force_numerical=True)

print(f"American Put Greeks:")
for greek, value in greeks.items():
    print(f"  {greek}: {value:.4f}")
```

### Example 3: Custom Strategy Selection

```python
# Force analytical even if product supports it
greeks_analytical = calculator.all_greeks(option, model, strategy_name="analytical")

# Force binomial with custom steps
greeks_binomial = calculator.all_greeks(
    option, 
    model, 
    strategy_name="binomial",
    n_steps=1000,
    use_richardson=False
)
```

### Example 4: Batch Calculation with Caching

```python
import numpy as np

strikes = np.linspace(80, 120, 20)
options = [
    EuropeanOption(S=100, K=K, T=30, option_type="call")
    for K in strikes
]

calculator = GreeksCalculator(enable_cache=True)

# First pass: cache misses (computes)
for opt in options:
    greeks = calculator.all_greeks(opt, model)

# Second pass: cache hits (instant)
for opt in options:
    greeks = calculator.all_greeks(opt, model)  # From cache
```

### Example 5: Greeks Surface (Delta vs Strike)

```python
import matplotlib.pyplot as plt

strikes = np.linspace(70, 130, 50)
deltas = []

for K in strikes:
    opt = EuropeanOption(S=100, K=K, T=30, option_type="call")
    greeks = calculator.all_greeks(opt, model)
    deltas.append(greeks['Delta'])

plt.plot(strikes, deltas)
plt.xlabel('Strike Price')
plt.ylabel('Delta')
plt.title('Delta vs Strike (Call Option)')
plt.grid(True)
plt.show()
```

### Example 6: Risk Analysis

```python
# Portfolio of options
portfolio = [
    EuropeanOption(S=100, K=95, T=30, option_type="call", qty=10),
    EuropeanOption(S=100, K=100, T=30, option_type="call", qty=5),
    EuropeanOption(S=100, K=105, T=30, option_type="put", qty=8),
]

# Aggregate Greeks
total_delta = 0
total_gamma = 0
total_vega = 0

for opt in portfolio:
    greeks = calculator.all_greeks(opt, model)
    total_delta += greeks['Delta']
    total_gamma += greeks['Gamma']
    total_vega += greeks['Vega']

print(f"Portfolio Delta: {total_delta:.2f}")
print(f"Portfolio Gamma: {total_gamma:.4f}")
print(f"Portfolio Vega: {total_vega:.2f}")

# Hedge recommendations
if abs(total_delta) > 0.5:
    print(f"Consider hedging: Delta exposure is {total_delta:.2f}")
```

## Parameter Tuning

### Binomial Steps (`n_steps`)

**Trade-off:** More steps = higher accuracy but slower computation.

| Use Case | Recommended `n_steps` |
|----------|----------------------|
| Quick estimates | 100-200 |
| Standard accuracy | 500-1000 |
| High precision | 2000-5000 |
| American options | 1000+ (early exercise needs fine grid) |

**Guideline:** Start with 500, increase if Greeks show instability.

### Richardson Extrapolation (`use_richardson`)

**What it does:** Computes at N and 2N steps, extrapolates to reduce discretization error.

**Advantages:**
- Improves accuracy without doubling final computation time
- Especially effective for Delta and Gamma (tree-extracted)

**When to disable:**
- Very tight performance requirements
- N is already very large (diminishing returns)

**Recommendation:** Keep enabled (`True`) unless profiling shows it's a bottleneck.

### Caching

**When enabled:** Results are cached based on product parameters, model parameters, and strategy options.

**Benefits:**
- Avoids recomputation in loops (sensitivity analysis, calibration)
- Significant speedup for repeated calculations

**Cache key includes:**
- Product: `S`, `K`, `T`, `option_type`, `qty`
- Model: `sigma`, `r`, `q` (or model-specific params)
- Strategy: name and options (`n_steps`, `use_richardson`)

**Memory:** Cache is unbounded by default (grows with distinct inputs). Consider `clear_cache()` periodically for long-running processes.

### Finite Difference Bumps

Binomial strategy uses these bump sizes (hardcoded but can be modified):
- **Spot bump**: `max(0.01, 0.1% * S)` for Delta/Gamma
- **Volatility bump**: `0.01` (1%) for Vega
- **Rate bump**: `1e-4` (1 basis point) for Rho
- **Time bump**: `1 day` for Theta

**Note:** Smaller bumps reduce truncation error but may increase numerical noise. Current defaults are industry-standard.

## Best Practices

1. **Use analytical when possible**: Faster and exact for European options with Black-Scholes
2. **Enable caching for batch operations**: Sensitivity analysis, calibration, portfolio valuation
3. **Tune `n_steps` based on accuracy needs**: Start conservative, increase if needed
4. **Keep Richardson enabled**: Good accuracy/performance trade-off
5. **Clear cache periodically**: For long-running processes with many unique inputs
6. **Validate against known values**: Compare analytical vs binomial for European options to verify setup

## Integration with Pricing Engines

GreeksCalculator is independent of pricing engines but uses them internally:
- **AnalyticalGreeksStrategy**: Uses `AnalyticalEngine` for price, then applies formulas
- **BinomialGreeksStrategy**: Uses `BinomialEngine` for pricing and tree extraction

This separation allows:
- Switching strategies without changing product/model code
- Extending to new strategies (e.g., Monte Carlo Greeks) by implementing `GreeksStrategy`
- Consistent API regardless of underlying calculation method

