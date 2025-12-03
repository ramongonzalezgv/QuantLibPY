### 1. Define Interfaces

Start by defining interfaces for the different components: models, valuation engines, and products. This will allow you to create different implementations without changing the core logic.

```python
from abc import ABC, abstractmethod

# Model Interface
class OptionModel(ABC):
    @abstractmethod
    def price(self, option_params):
        pass

# Valuation Engine Interface
class ValuationEngine(ABC):
    @abstractmethod
    def value(self, model: OptionModel, option_params):
        pass

# Product Interface
class OptionProduct(ABC):
    @abstractmethod
    def get_parameters(self):
        pass
```

### 2. Implement Models

Create concrete implementations of the `OptionModel` interface for each pricing model (e.g., Black-Scholes, Heston).

```python
class BlackScholesModel(OptionModel):
    def price(self, option_params):
        # Implement Black-Scholes pricing logic
        pass

class HestonModel(OptionModel):
    def price(self, option_params):
        # Implement Heston pricing logic
        pass
```

### 3. Implement Valuation Engines

Create concrete implementations of the `ValuationEngine` interface for each valuation method (e.g., analytical, FFT, Monte Carlo).

```python
class AnalyticalValuationEngine(ValuationEngine):
    def value(self, model: OptionModel, option_params):
        return model.price(option_params)

class FFTValuationEngine(ValuationEngine):
    def value(self, model: OptionModel, option_params):
        # Implement FFT valuation logic
        pass

class MonteCarloValuationEngine(ValuationEngine):
    def value(self, model: OptionModel, option_params):
        # Implement Monte Carlo valuation logic
        pass
```

### 4. Implement Products

Create concrete implementations of the `OptionProduct` interface for different types of options (e.g., vanilla options, exotic options).

```python
class VanillaOption(OptionProduct):
    def __init__(self, strike, expiry):
        self.strike = strike
        self.expiry = expiry

    def get_parameters(self):
        return {'strike': self.strike, 'expiry': self.expiry}

class ExoticOption(OptionProduct):
    def __init__(self, strike, expiry, barrier_type):
        self.strike = strike
        self.expiry = expiry
        self.barrier_type = barrier_type

    def get_parameters(self):
        return {'strike': self.strike, 'expiry': self.expiry, 'barrier_type': self.barrier_type}
```

### 5. Create a Valuation Context

You can create a context or a factory that takes a model, a valuation engine, and a product to perform the valuation.

```python
class OptionValuationContext:
    def __init__(self, engine: ValuationEngine):
        self.engine = engine

    def value_option(self, model: OptionModel, product: OptionProduct):
        option_params = product.get_parameters()
        return self.engine.value(model, option_params)
```

### 6. Usage Example

Now you can easily combine different models, engines, and products:

```python
# Create instances of models, engines, and products
bs_model = BlackScholesModel()
heston_model = HestonModel()

analytical_engine = AnalyticalValuationEngine()
fft_engine = FFTValuationEngine()

vanilla_option = VanillaOption(strike=100, expiry=1)
exotic_option = ExoticOption(strike=100, expiry=1, barrier_type='knock-in')

# Valuation
valuation_context = OptionValuationContext(analytical_engine)
bs_price = valuation_context.value_option(bs_model, vanilla_option)
heston_price = valuation_context.value_option(heston_model, exotic_option)

print(f"Black-Scholes Price: {bs_price}")
print(f"Heston Price: {heston_price}")
```

### Conclusion

This design allows you to easily add new models, valuation engines, and products by simply creating new classes that implement the respective interfaces. You can also mix and match different components without changing the underlying logic, making your codebase more maintainable and scalable.