// ...existing code...

# OptionPricingPY

![Badge en Desarollo](https://img.shields.io/badge/STATUS-BETA-green)  
![Python](https://img.shields.io/badge/Python-3.11.4-blue)

Lightweight, extensible framework for option pricing and valuation engines. Architecture and full technical details live in the docs/ folder.

## What this repo contains
- Core interfaces and example implementations for:
  - Models (e.g., Black-Scholes, Heston)
  - Valuation engines (analytical, Monte Carlo, FFT)
  - Products (vanilla options, etc.)
- Tests, examples, and documentation under docs/

See docs/architecture.md and docs/valuation_context.md for the detailed design and rationale.

## Schematic overview of main classes

- OptionModel (interface)
  - price(option_parameters) -> float

- ValuationEngine (interface)
  - value(model: OptionModel, option_parameters) -> float

- OptionProduct (interface)
  - get_parameters() -> dict

Concrete examples:
- BlackScholesModel : OptionModel
- HestonModel : OptionModel
- AnalyticalValuationEngine : ValuationEngine
- MonteCarloValuationEngine : ValuationEngine
- VanillaOption : OptionProduct

Relationship (conceptual):
- OptionProduct -> provides parameters -> ValuationEngine
- ValuationEngine -> uses OptionModel to compute value
- OptionValuationContext -> orchestrates engine + model + product

Simple ASCII diagram:
```
[OptionProduct] --get_parameters()--> {params}
                                   |
                                   v
[OptionValuationContext] --calls--> [ValuationEngine] --uses--> [OptionModel]
                                   |
                                   v
                                price (float)
```

## Minimal usage examples

Python (quick example):
```python
from option_pricing.models import BlackScholesModel
from option_pricing.engines import AnalyticalValuationEngine
from option_pricing.products import VanillaOption
from option_pricing.context import OptionValuationContext

bs = BlackScholesModel()
engine = AnalyticalValuationEngine()
product = VanillaOption(strike=100, maturity=1.0, option_type='call')

context = OptionValuationContext(engine)
price = context.value_option(bs, product)
print("Vanilla call price:", price)
```

Simple Monte Carlo sketch:
```python
from option_pricing.models import BlackScholesModel
from option_pricing.engines import MonteCarloValuationEngine

mc_engine = MonteCarloValuationEngine(n_paths=100_000, seed=42)
price = mc_engine.value(BlackScholesModel(), product.get_parameters())
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
