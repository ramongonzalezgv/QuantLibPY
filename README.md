# OptionPricingPY

![Badge en Desarollo](https://img.shields.io/badge/Status-Beta-green) ![Python](https://img.shields.io/badge/Python-3.11.4-blue) ![Badge versiom](https://img.shields.io/badge/version-v.0.2.0-green) ![Python](https://img.shields.io/badge/Python-3.11.4-blue)

Lightweight, extensible framework for option pricing and valuation engines. Full technical details live in the docs/ folder (see docs/architecture.md and docs/valuation_context.md).

## Overview

This project separates three responsibilities:

- Models — encapsulate asset dynamics and pricing primitives.
- Engines — implement valuation algorithms (analytical, Monte Carlo, FFT).
- Products — describe option contract terms and convert them to canonical parameters.

An OptionValuationContext orchestrates the three so users can swap models/engines/products without changing orchestration code.

## Visual class examples

Below are minimal, illustrative Python snippets to show structure and interactions. These are deliberately small; full implementations live in the source package.

### Interfaces (schematic)

```python
class OptionModel:
    """Model interface: pricing primitives and simulators."""
    def price(self, params: dict) -> float:
        raise NotImplementedError

    def simulate_paths(self, params: dict, n_paths: int):
        raise NotImplementedError

    def characteristic_function(self, u, params: dict):
        raise NotImplementedError


class ValuationEngine:
    """Engine interface: given a model and product params, return price (and optionally diagnostics)."""
    def value(self, model: OptionModel, params: dict):
        raise NotImplementedError


class OptionProduct:
    """Product interface: canonicalize contract terms into a params dict."""
    def get_parameters(self) -> dict:
        raise NotImplementedError
```

### Concrete examples (compact)

```python
class BlackScholesModel(OptionModel):
    def __init__(self, spot: float, vol: float, rate: float):
        self.spot = spot
        self.vol = vol
        self.rate = rate

    def price(self, params):
        # closed-form Black-Scholes formula (placeholder)
        K = params["strike"]
        T = params["maturity"]
        option_type = params.get("option_type", "call")
        # ...compute analytic price...
        return 10.0  # example


class AnalyticalValuationEngine(ValuationEngine):
    def value(self, model: OptionModel, params: dict):
        # For analytic engines, delegate to model.price
        return model.price(params)


class MonteCarloValuationEngine(ValuationEngine):
    def __init__(self, n_paths: int = 10000, seed: int | None = None):
        self.n_paths = n_paths
        self.seed = seed

    def value(self, model: OptionModel, params: dict):
        # Use model.simulate_paths to estimate expectation (placeholder)
        paths = model.simulate_paths(params, self.n_paths)
        # ...discount payoff average...
        return 9.8  # example
```

### Product example

```python
class VanillaOption(OptionProduct):
    def __init__(self, strike: float, maturity: float, option_type: str = "call"):
        self.strike = strike
        self.maturity = maturity
        self.option_type = option_type

    def get_parameters(self) -> dict:
        return {
            "strike": self.strike,
            "maturity": self.maturity,
            "option_type": self.option_type
        }
```

### Orchestration: OptionValuationContext

```python
class OptionValuationContext:
    """
    Orchestrates product -> engine -> model.
    The context takes a ValuationEngine (which can carry configuration)
    and provides a simple API to value products with a model.
    """
    def __init__(self, engine: ValuationEngine):
        self.engine = engine

    def value_option(self, model: OptionModel, product: OptionProduct):
        params = product.get_parameters()
        # Optionally enrich params here (e.g., attach model market data)
        return self.engine.value(model, params)
```

## Usage examples

Minimal, runnable-style example (adjust import paths for your package layout):

```python
# create product
product = VanillaOption(strike=100, maturity=1.0, option_type="call")

# create model (holds market data)
bs_model = BlackScholesModel(spot=100.0, vol=0.2, rate=0.01)

# choose engine and context
engine = AnalyticalValuationEngine()
context = OptionValuationContext(engine)

# value
price = context.value_option(bs_model, product)
print("Vanilla call price:", price)
```

Monte Carlo sketch:

```python
mc_engine = MonteCarloValuationEngine(n_paths=100_000, seed=42)
context = OptionValuationContext(mc_engine)
price = context.value_option(bs_model, product)
print("MC estimate:", price)
```

(Concrete import paths may vary — check the package layout in the repository.)

## Quick start (Windows)
1. Clone:
```powershell
git clone https://github.com/ramongonzalezgv/OptionPricingPY.git "c:\Users\<your_username>\Desktop\OptionPricingPY"
cd "c:\Users\<your_username>\Desktop\OptionPricingPY"
```
2. Create & activate venv:
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1   # PowerShell
# or
.venv\Scripts\activate.bat   # cmd.exe
```
3. Install deps:
```powershell
pip install -r requirements.txt
```
4. Run tests:
```powershell
pytest
```

## Project layout (high-level)
- docs/ — architecture.md, valuation_context.md, and other docs
- option_pricing/ or src/ — core package (models, engines, products, context)
- examples/ — runnable demos
- tests/ — unit tests
- requirements.txt, LICENSE, README.md

## Documentation
Primary docs:
- docs/architecture.md — overall architecture and interfaces
- docs/valuation_context.md — how context and orchestration work
- docs/* — implementation notes and API details

## Contributing
- Fork, create branch, open PR
- Add tests for new features/bugs
- Update docs under docs/

## License
See LICENSE in the repository root.

## Contact
Open an issue for bugs, feature requests, or questions.
