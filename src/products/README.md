# FinancialProduct: Implements different financial instruments

## Table of Contents
1. [FinancialProduct class](#introduction)
2. [Architecture](#architecture)
3. [How Caching Works](#how-caching-works)
4. [How OptionValuationContext Works](#how-optionvaluationcontext-works)
5. [Usage Examples](#usage-examples)
6. [Advantages and Use Cases](#advantages-and-use-cases)

## FinancialProduct class
```python
class FinancialProduct(ABC):
    """Base class for financial products."""
    
    @abstractmethod
    def payoff(self, spot_prices: np.ndarray) -> np.ndarray:
        """Calculates the payoff of the product."""
        pass

    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """
        Mandatory method to return essential parameters 
        of the contract (K, T, etc.) for the cache key.
        """
        pass
```

## EuropeanOption
```python
class EuropeanOption(FinancialProduct):
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