# QuantLibPY

![Badge en Desarollo](https://img.shields.io/badge/Status-Beta-yellow) ![Badge versio](https://img.shields.io/badge/version-v.0.5.0-green) ![Python](https://img.shields.io/badge/Python-3.11.4-blue)

Lightweight, extensible framework for pricing options with swappable products, models, and engines. Designed for experimentation (analytical, FFT, binomial, Monte Carlo) with a simple orchestration context, caching, and optional parallelism.

## Quick links
- High-level docs: [architecture.md](docs/architecture.md), [valuationContext_ENG.md](docs/valuationContext_ENG.md)
- Product details: [src/products/README.md](src/products/README.md)
- Model details: [src/models/README.md](src/models/README.md)
- Engine details: [src/engines/README.md](src/engines/README.md)
- Valuation context: [src/valuation/README.md](src/valuation/README.md)
- Greeks calculation: [src/greeks/README.md](src/greeks/README.md)

## What's inside
- **Products**: European, American, Asian options implement payoffs and contract parameters.
- **Models**: Black-Scholes, Black (76), Heston provide characteristic functions and helpers.
- **Engines**: Analytical, Binomial, FFT, Monte Carlo implement pricing schemes.
- **Context**: `OptionValuationContext` coordinates product + model + engine with optional caching and parallel batch pricing.
- **Greeks**: Automatic calculation of Delta, Gamma, Theta, Vega, Rho with analytical and numerical strategies.
- **Notebooks**: runnable examples under `notebooks/`.

## Minimal usage

After installation (see [Installation](#installation)), you can use the package:

```python
from src.products.EuropeanOption import EuropeanOption
from src.models.BlackScholesModel import BlackScholesModel
from src.engines.AnalyticalEngine import AnalyticalEngine
from src.valuation.OptionValuationContext import OptionValuationContext

option = EuropeanOption(S=100, K=100, T=30, option_type="call", qty=1)
model = BlackScholesModel(sigma=0.2, r=0.01, q=0.0)
engine = AnalyticalEngine()

ctx = OptionValuationContext(engine, cache_enabled=True)
price = ctx.value_option(option, model)
print(f"Call price: {price:.4f}")
```

**FFT example (Heston):**
```python
from src.products.EuropeanOption import EuropeanOption
from src.models.HestonModel import HestonModel
from src.engines.FFTEngine import FFTEngine
from src.valuation.OptionValuationContext import OptionValuationContext

option = EuropeanOption(S=100, K=95, T=180, option_type="call", qty=1)
model = HestonModel(kappa=2.0, theta=0.04, sigma=0.6, rho=-0.7, v0=0.04, r=0.01, q=0.0)
engine = FFTEngine(N=2**12, B=200)

ctx = OptionValuationContext(engine)
price = ctx.value_option(option, model)
```

**Monte Carlo (Asian):**
```python
from src.products.AsianOption import AsianOption
from src.models.BlackScholesModel import BlackScholesModel
from src.engines.MonteCarloEngine import MonteCarloEngine
from src.valuation.OptionValuationContext import OptionValuationContext

option = AsianOption(S=100, K=100, T=365, option_type="call", averaging_type="arithmetic")
model = BlackScholesModel(sigma=0.25, r=0.015, q=0.0)
engine = MonteCarloEngine(n_paths=5000, n_steps=252, seed=42)

ctx = OptionValuationContext(engine, cache_enabled=False)
price = ctx.value_option(option, model)
```

## Installation

This package is currently available directly from GitHub (not yet on PyPI). To use it:

### Requirements

- Python 3.8 or higher (tested with Python 3.11.4)
- Dependencies: `numpy>=1.21`, `scipy>=1.8`

### Option 1: Install in development mode (recommended)

This allows you to import the package as if it were installed, and changes to the code are immediately available.

1. **Clone the repository:**
```bash
git clone https://github.com/ramongonzalezgv/OptionPricingPY.git
cd OptionPricingPY
```

2. **Create and activate a virtual environment:**
```bash
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

# Windows (cmd.exe)
python -m venv .venv
.venv\Scripts\activate.bat

# Linux/Mac
python -m venv .venv
source .venv/bin/activate
```

3. **Install the package in development mode:**
```bash
pip install -e .
```

This will automatically install all dependencies from `requirements.txt` and make the package importable.

### Option 2: Install dependencies only

If you prefer to work with the source code directly without installing the package:

1. Clone the repository (same as above)
2. Create and activate virtual environment (same as above)
3. Install dependencies:
```bash
pip install -r requirements.txt
```

Then add the `src` directory to your Python path or use relative imports in your scripts.

## Quick start

After installation (see [Installation](#installation) above), see the [Minimal usage](#minimal-usage) section for code examples.

### Run tests

```bash
pytest
```

### Explore examples

Check out the Jupyter notebooks in `notebooks/tests/` for more detailed examples and use cases.

## Project layout (high-level)
- `docs/` — architecture and valuation notes
- `src/products/` — option contracts ([README](src/products/README.md))
- `src/models/` — stochastic models ([README](src/models/README.md))
- `src/engines/` — pricing engines ([README](src/engines/README.md))
- `src/valuation/` — orchestration/context ([README](src/valuation/README.md))
- `src/greeks/` — sensitivity calculations ([README](src/greeks/README.md))

## Contributing
- Fork, create branch, open PR
- Add tests for new features/bugs
- Update docs in `docs/` and relevant module READMEs

## License
See LICENSE in the repository root.

## Contact
Open an issue for bugs, feature requests, or questions.
