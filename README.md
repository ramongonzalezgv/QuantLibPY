# OptionPricingPY

![Badge en Desarollo](https://img.shields.io/badge/STATUS-BETA-green)
![Python](https://img.shields.io/badge/Python-3.11.4-blue)

Lightweight, extensible framework for option pricing and valuation engines.  
This repository provides interfaces and concrete implementations for option models, valuation engines, and product definitions, with architecture and technical details stored in the docs/ folder.

## Key ideas
- Separation of concerns: models, valuation engines, and products are decoupled via simple interfaces.
- Extensible: add new models or valuation engines by implementing the provided interfaces.
- Documentation-first: design and technical details live under `docs/`.

## Features
- Interface definitions for models, engines, and products
- Example concrete models (e.g., Black-Scholes)
- Multiple valuation engine patterns (analytical, Monte Carlo, FFT)
- Pluggable product types (vanilla options, etc.)
- Tests and examples (see corresponding folders)

## Quick start (Windows)
1. Clone the repo
```bash
git clone <repo-url> "c:\Users\ramon\OneDrive\Desktop\Python projects\OptionPricingPY"
cd "c:\Users\ramon\OneDrive\Desktop\Python projects\OptionPricingPY"
```

2. Create and activate a virtual environment
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1   # PowerShell
# or
.venv\Scripts\activate.bat   # Command Prompt
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Run tests
```bash
pytest
```

5. Run examples / demos  
Check the `examples/` or `scripts/` directory for runnable demos. Example:
```bash
python examples/demo.py
```

## Project layout (high-level)
- docs/ — architecture and detailed technical docs (see docs/architecture.md)
- option_pricing/ or src/ — core package (models, engines, products)
- examples/ — runnable examples and small demos
- tests/ — unit tests
- requirements.txt — project dependencies
- README.md — this file

(Actual package/module names and exact locations are in the repo — adapt the commands above accordingly.)

## Documentation
All detailed design, API descriptions, and implementation notes are in the docs/ folder:
- docs/architecture.md — architecture overview and interfaces
- docs/* — additional design and usage docs

## Contributing
- Fork -> feature branch -> open PR
- Add tests for new features or bug fixes
- Keep changes small and focused; update docs under `docs/` as needed

## License
See the LICENSE file in the repository root for license details.

## Contact / Support
Open an issue in the repository for bugs, feature requests, or questions.
