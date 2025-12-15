# SABR Model Validation Tests

This directory contains validation tests for the SABR model implementation.

## test_sabr_validation.ipynb

This Jupyter notebook validates our SABR implementation against the `pysabr` library to ensure correctness.

### What it tests:

1. **Lognormal SABR (β=1) Validation**
   - Compares implied volatilities with pysabr
   - Tests various strikes (ATM, ITM, OTM)
   - Visualizes volatility smiles
   - Compares option prices

2. **Normal SABR (β=0) Validation**
   - Tests our Normal SABR implementation
   - Visualizes volatility smiles
   - Compares with Lognormal SABR

3. **Multiple Parameter Sets**
   - Tests different combinations of alpha, rho, and nu
   - Validates robustness across parameter space

4. **Integration Tests**
   - Verifies SABR models work with AnalyticalEngine
   - Tests end-to-end pricing workflow

5. **Edge Cases**
   - ATM options (F = K)
   - Deep ITM/OTM options
   - Boundary conditions

### Requirements

To run this notebook, you'll need:

```bash
pip install pysabr matplotlib pandas jupyter
```

Or install from requirements.txt (pysabr is optional):

```bash
pip install -r requirements.txt
pip install pysabr matplotlib pandas jupyter
```

### Running the Notebook

1. Start Jupyter:
   ```bash
   jupyter notebook
   ```

2. Navigate to `notebooks/tests/test_sabr_validation.ipynb`

3. Run all cells to execute the validation tests

### Expected Results

The notebook will:
- Display comparison tables showing differences between our implementation and pysabr
- Generate volatility smile plots
- Show that differences are minimal (typically < 0.01% for lognormal SABR)
- Validate that our implementation is correct

### Notes

- pysabr primarily supports lognormal SABR (β=1)
- For Normal SABR (β=0), we validate our implementation internally
- The notebook will automatically install pysabr if it's not available
