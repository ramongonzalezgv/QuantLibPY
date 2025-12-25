from typing import Optional
from dataclasses import dataclass

@dataclass
class MarketQuote:
    """
    Represents a single market quote (option price or implied volatility).
    """
    strike: float
    maturity: float
    market_value: float  # Can be price or implied volatility
    quote_type: str = "volatility"  # "volatility" or "price"
    forward: Optional[float] = None
    weight: float = 1.0
    bid: Optional[float] = None
    ask: Optional[float] = None
    
    @property
    def mid(self) -> float:
        """Return market value (or mid if bid/ask provided)."""
        if self.bid is not None and self.ask is not None:
            return (self.bid + self.ask) / 2
        return self.market_value
    
    @property
    def spread(self) -> Optional[float]:
        """Return bid-ask spread if available."""
        if self.bid is not None and self.ask is not None:
            return self.ask - self.bid
        return None