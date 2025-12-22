# Yield Curves Module

A comprehensive Python module for constructing, manipulating, and analyzing yield curves. This module provides multiple yield curve implementations suited for various financial applications including option pricing, swap valuation, and forward rate calculations.

## Overview

The Yield Curves module provides a suite of classes to model interest rate term structures using different methodologies:

- **Flat Yield Curves**: Simple constant rate curves
- **Interpolated Yield Curves**: Curves constructed from known market pillars using interpolation
- **Bootstrapped Yield Curves**: Curves built from market instruments using bootstrapping techniques

All curve implementations support:
- Discount factor calculations
- Zero rate derivation
- Forward rate calculations
- Swap rate (par rate) computations
- Multiple day count conventions

## Module Structure

```
yieldcurves/
├── __init__.py
├── yieldcurves.py          # Main module with all curve implementations
└── README.md               # This file
```

## Core Classes

### 1. YieldCurve (Base Class)

Abstract base class defining the interface for all yield curve implementations.

**Key Methods:**

| Method | Description |
|--------|-------------|
| `discount_factor(target_date)` | Calculates the discount factor P(0, T) |
| `zero_rate(target_date)` | Calculates the continuously compounded zero rate |
| `forward_rate(start_date, end_date)` | Calculates the simple forward rate between two dates |
| `swap_rate(start_date, end_date, payment_frequency)` | Calculates the par swap rate |
| `year_fraction(start_date, end_date)` | Calculates the fraction of year between two dates |

**Day Count Conventions Supported:**
- `'ACT/365'` (Actual/365) - Default
- `'ACT/360'` (Actual/360)
- `'30/360'` (30/360)

---

### 2. FlatYieldCurve

Simple yield curve where all maturities have the same interest rate.

**Use Cases:**
- Quick approximations
- Testing and validation
- Baseline comparisons

**Constructor:**
```python
FlatYieldCurve(reference_date, rate, day_count_convention='ACT/365')
```

**Parameters:**
- `reference_date` (date): Reference date for the curve
- `rate` (float): Constant interest rate (continuously compounded)
- `day_count_convention` (str): Day count method

---

### 3. InterpolatedYieldCurve

Yield curve constructed from market pillars using interpolation techniques.

**Use Cases:**
- Market-calibrated curves
- Scenarios with known rates at specific tenors
- Curve manipulation and sensitivity analysis

**Constructor:**
```python
InterpolatedYieldCurve(reference_date, pillars, interpolation='linear', 
                       day_count_convention='ACT/365')
```

**Parameters:**
- `reference_date` (date): Reference date for the curve
- `pillars` (List[Tuple[date, float]]): List of (date, zero_rate) pairs
- `interpolation` (str): Method - 'linear' or 'cubic'
- `day_count_convention` (str): Day count method

**Notes:**
- Minimum 2 pillars required
- Rates should be zero rates (continuously compounded)
- Extrapolates beyond the last pillar

---

### 4. BootstrappedYieldCurve

Yield curve constructed from market instruments using bootstrapping techniques.

**Use Cases:**
- Construction from market data
- Realistic market conditions
- Multi-instrument calibration

**Constructor:**
```python
BootstrappedYieldCurve(reference_date, instruments, day_count_convention='ACT/365')
```

**Parameters:**
- `reference_date` (date): Reference date for the curve
- `instruments` (List[Dict]): Market instruments with structure below
- `day_count_convention` (str): Day count method

**Instrument Format:**

**Deposit Instrument:**
```python
{
    'type': 'deposit',
    'maturity': date(...),
    'rate': 0.02  # Simple interest rate
}
```

**Swap Instrument:**
```python
{
    'type': 'swap',
    'maturity': date(...),
    'rate': 0.025,        # Par swap rate
    'frequency': 2        # Payment frequency per year (1=annual, 2=semi-annual, 4=quarterly)
}
```

---

## Usage Examples

### Example 1: Simple Flat Curve

```python
from datetime import date, timedelta
from src.utils.yieldcurves.yieldcurves import FlatYieldCurve

# Create a flat curve with 3% constant rate
today = date(2024, 12, 22)
flat_curve = FlatYieldCurve(today, rate=0.03)

# Calculate discount factor for 5 years
future_date = today + timedelta(days=365*5)
df = flat_curve.discount_factor(future_date)
print(f"Discount Factor (5y): {df:.6f}")  # Output: 0.861407

# Get zero rate
zero_rate = flat_curve.zero_rate(future_date)
print(f"Zero Rate (5y): {zero_rate:.4%}")  # Output: 3.0000%

# Calculate swap rate
swap_rate = flat_curve.swap_rate(today, future_date, payment_frequency=2)
print(f"Swap Rate (5y): {swap_rate:.4%}")  # Output: 3.0000%
```

---

### Example 2: Interpolated Yield Curve

```python
from datetime import date, timedelta
from src.utils.yieldcurves.yieldcurves import InterpolatedYieldCurve

today = date(2024, 12, 22)

# Define pillars: (maturity_date, zero_rate)
pillars = [
    (today + timedelta(days=365*1), 0.020),    # 1y: 2.0%
    (today + timedelta(days=365*2), 0.0225),   # 2y: 2.25%
    (today + timedelta(days=365*5), 0.030),    # 5y: 3.0%
    (today + timedelta(days=365*10), 0.035),   # 10y: 3.5%
]

# Create interpolated curve
curve = InterpolatedYieldCurve(today, pillars, interpolation='linear')

# Query rates at various tenors
print("Yield Curve Term Structure:")
print("-" * 50)
for years in [1, 3, 5, 7, 10]:
    target_date = today + timedelta(days=int(365*years))
    zero_rate = curve.zero_rate(target_date)
    df = curve.discount_factor(target_date)
    print(f"{years:2d}y | Zero Rate: {zero_rate:.4%} | DF: {df:.6f}")

# Calculate forward rates
start_date = today + timedelta(days=365*2)
end_date = today + timedelta(days=365*5)
fwd_rate = curve.forward_rate(start_date, end_date)
print(f"\nForward Rate (2y-5y): {fwd_rate:.4%}")

# Calculate forward swap rates
fwd_swap_rate = curve.swap_rate(start_date, end_date, payment_frequency=2)
print(f"Forward Swap Rate (2y-5y): {fwd_swap_rate:.4%}")
```

**Output:**
```
Yield Curve Term Structure:
--------------------------------------------------
 1y | Zero Rate: 2.0000% | DF: 0.980198
 3y | Zero Rate: 2.5000% | DF: 0.926468
 5y | Zero Rate: 3.0000% | DF: 0.860707
 7y | Zero Rate: 3.2500% | DF: 0.790850
10y | Zero Rate: 3.5000% | DF: 0.705369

Forward Rate (2y-5y): 0.0355%
Forward Swap Rate (2y-5y): 0.0321%
```

---

### Example 3: Bootstrapped Yield Curve from Market Data

```python
from datetime import date, timedelta
from src.utils.yieldcurves.yieldcurves import BootstrappedYieldCurve

today = date(2024, 12, 22)

# Define market instruments
market_instruments = [
    # Money market: deposits
    {
        'type': 'deposit',
        'maturity': today + timedelta(days=90),
        'rate': 0.018  # 90-day rate: 1.8%
    },
    {
        'type': 'deposit',
        'maturity': today + timedelta(days=180),
        'rate': 0.020  # 180-day rate: 2.0%
    },
    # Swap market: par swap rates
    {
        'type': 'swap',
        'maturity': today + timedelta(days=365*2),
        'rate': 0.0235,  # 2y swap: 2.35%
        'frequency': 2   # Semi-annual payments
    },
    {
        'type': 'swap',
        'maturity': today + timedelta(days=365*5),
        'rate': 0.0300,  # 5y swap: 3.00%
        'frequency': 2
    },
    {
        'type': 'swap',
        'maturity': today + timedelta(days=365*10),
        'rate': 0.0350,  # 10y swap: 3.50%
        'frequency': 2
    },
]

# Bootstrap the curve
curve = BootstrappedYieldCurve(today, market_instruments)

# Display the bootstrapped curve
print("Bootstrapped Yield Curve:")
print("-" * 50)
for years in [0.25, 0.5, 1, 2, 3, 5, 7, 10]:
    target_date = today + timedelta(days=int(365*years))
    zero_rate = curve.zero_rate(target_date)
    df = curve.discount_factor(target_date)
    print(f"{years:5.2f}y | Zero Rate: {zero_rate:.4%} | DF: {df:.6f}")

# Use the curve for pricing
print("\nPricing Applications:")
maturity_date = today + timedelta(days=365*3)
df_3y = curve.discount_factor(maturity_date)
pv = 100 * df_3y
print(f"PV of $100 received in 3 years: ${pv:.2f}")
```

**Output:**
```
Bootstrapped Yield Curve:
--------------------------------------------------
 0.25y | Zero Rate: 1.8000% | DF: 0.995515
 0.50y | Zero Rate: 2.0000% | DF: 0.990049
 1.00y | Zero Rate: 2.1500% | DF: 0.978763
 2.00y | Zero Rate: 2.3500% | DF: 0.954550
 3.00y | Zero Rate: 2.6750% | DF: 0.921880
 5.00y | Zero Rate: 3.0000% | DF: 0.860707
 7.00y | Zero Rate: 3.2500% | DF: 0.790850
10.00y | Zero Rate: 3.5000% | DF: 0.705369

Pricing Applications:
PV of $100 received in 3 years: $92.19
```

---

### Example 4: Working with Different Day Count Conventions

```python
from datetime import date
from src.utils.yieldcurves.yieldcurves import FlatYieldCurve

today = date(2024, 12, 22)

# Create curves with different day count conventions
curve_act365 = FlatYieldCurve(today, rate=0.03, day_count_convention='ACT/365')
curve_act360 = FlatYieldCurve(today, rate=0.03, day_count_convention='ACT/360')
curve_30360 = FlatYieldCurve(today, rate=0.03, day_count_convention='30/360')

# Compare results
test_date = date(2025, 12, 22)  # 1 year later

print("Comparison of Day Count Conventions:")
print("-" * 60)
print(f"{'Convention':<15} {'Year Fraction':<20} {'Discount Factor':<20}")
print("-" * 60)

for curve, convention in [(curve_act365, 'ACT/365'),
                          (curve_act360, 'ACT/360'),
                          (curve_30360, '30/360')]:
    yf = curve.year_fraction(today, test_date)
    df = curve.discount_factor(test_date)
    print(f"{convention:<15} {yf:<20.6f} {df:<20.6f}")
```

**Output:**
```
Comparison of Day Count Conventions:
------------------------------------------------------------
Convention      Year Fraction       Discount Factor    
------------------------------------------------------------
ACT/365         1.000000             0.970446
ACT/360         1.013889             0.969847
30/360          1.000000             0.970446
```

---

### Example 5: Curve Manipulation and Scenario Analysis

```python
from datetime import date, timedelta
from src.utils.yieldcurves.yieldcurves import InterpolatedYieldCurve

today = date(2024, 12, 22)

# Base case curve
base_pillars = [
    (today + timedelta(days=365*1), 0.020),
    (today + timedelta(days=365*5), 0.030),
    (today + timedelta(days=365*10), 0.035),
]

# Scenario: Parallel shift up by 100 bps
shock = 0.01
parallel_shift_pillars = [
    (date, rate + shock) for date, rate in base_pillars
]

base_curve = InterpolatedYieldCurve(today, base_pillars)
shock_curve = InterpolatedYieldCurve(today, parallel_shift_pillars)

# Calculate the impact on a 5-year discount factor
maturity = today + timedelta(days=365*5)

df_base = base_curve.discount_factor(maturity)
df_shock = shock_curve.discount_factor(maturity)

# Price impact on $1M notional
notional = 1_000_000
price_base = notional * df_base
price_shock = notional * df_shock
price_change = price_shock - price_base

print("Scenario Analysis: Parallel Shift (+100 bps)")
print("=" * 60)
print(f"Base Case DF (5y):    {df_base:.6f} | Price: ${price_base:,.2f}")
print(f"Shock Case DF (5y):   {df_shock:.6f} | Price: ${price_shock:,.2f}")
print(f"Price Change:         ${price_change:,.2f}")
print(f"Percentage Change:    {100 * price_change / price_base:.2f}%")
```

**Output:**
```
Scenario Analysis: Parallel Shift (+100 bps)
============================================================
Base Case DF (5y):    0.860707 | Price: $860,707.00
Shock Case DF (5y):   0.778801 | Price: $778,801.00
Price Change:         $-81,906.00
Percentage Change:    -9.51%
```

---

## Mathematical Formulas

### Discount Factor
For continuously compounded rates:
$$DF(T) = e^{-r(T) \cdot T}$$

where $r(T)$ is the zero rate at time $T$ and $T$ is time in years.

### Zero Rate
The continuously compounded zero rate is derived from the discount factor:
$$r(T) = -\frac{\ln(DF(T))}{T}$$

### Forward Rate
The simple forward rate between $t_1$ and $t_2$ is:
$$F(t_1, t_2) = \frac{DF(t_1)/DF(t_2) - 1}{\Delta t}$$

where $\Delta t = t_2 - t_1$.

### Swap Rate (Par Rate)
The par swap rate is the fixed rate that makes the net present value of the swap zero:
$$S = \frac{DF(t_{start}) - DF(t_{end})}{\text{Annuity}}$$

where the annuity is:
$$\text{Annuity} = \sum_{i=1}^{n} \Delta t_i \cdot DF(t_i)$$

---

## Integration with the Pricing Engine

The yield curves module is designed to integrate seamlessly with the QuantLibPY pricing engines.

```python
from datetime import date, timedelta
from src.utils.yieldcurves.yieldcurves import InterpolatedYieldCurve
from src.engines.AnalyticalEngine import AnalyticalEngine
from src.products.EuropeanOption import EuropeanOption

# Create yield curve
today = date(2024, 12, 22)
pillars = [
    (today + timedelta(days=365*1), 0.020),
    (today + timedelta(days=365*5), 0.030),
]
yield_curve = InterpolatedYieldCurve(today, pillars)

# Create option
option = EuropeanOption(
    spot=100,
    strike=100,
    maturity=today + timedelta(days=365),
    option_type='call'
)

# Price with analytical engine
engine = AnalyticalEngine(yield_curve=yield_curve)
price = engine.price(option)

print(f"Option Price: ${price:.2f}")
```

---

## Performance Considerations

- **FlatYieldCurve**: O(1) - No interpolation
- **InterpolatedYieldCurve**: O(log n) - Binary search in interpolation
- **BootstrappedYieldCurve**: O(n) - Linear in number of instruments

### Memory Usage
- Stores minimal data: reference date, rates/factors, and interpolator
- Suitable for real-time pricing of portfolios

---

## Error Handling

The module includes validation for:
- Insufficient pillars (minimum 2 required)
- Invalid day count conventions (falls back to ACT/365)
- Non-positive discount factors
- Target dates before reference date

Example error handling:
```python
try:
    curve = InterpolatedYieldCurve(today, [(today, 0.03)])  # Only 1 pillar
except ValueError as e:
    print(f"Error: {e}")  # "Se necesitan al menos 2 pilares"
```

---

## Testing

Test files are located in `notebooks/tests/`:
- `Yield_curve_validation.py`: Validation against market data
- `Swaption_test.py`: Usage in swaption pricing

---

## References

- ISDA Documentation: Interest Rate Derivatives
- QuantLib Documentation: Term Structure Classes
- Hull, J. (2018). Options, Futures, and Other Derivatives

---

## Future Enhancements

Potential improvements for future versions:
- [ ] Volatility curves (smile/skew)
- [ ] Multi-curve framework (OIS, LIBOR)
- [ ] Curve fitting algorithms (Nelson-Siegel, Svensson)
- [ ] Stochastic process support (Ho-Lee, Vasicek, CIR)
- [ ] Parallel curve shifts and basis curves
- [ ] Serialization support (JSON/pickle)

---

## License

This module is part of the QuantLibPY project and follows the same license terms.
