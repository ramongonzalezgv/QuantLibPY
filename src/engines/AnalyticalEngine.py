from ..models.BlackScholesModel import BlackScholesModel
from ..models.BlackModel import BlackModel
from ..models.SABRModel import SABRModel, NormalSABRModel, LognormalSABRModel
import numpy as np
import scipy.stats as si


from .PricingEngine import PricingEngine

class AnalyticalEngine(PricingEngine):
    """Motor analítico para fórmulas cerradas (Black-Scholes, Black, SABR)."""
    
    def calculate_price(self, product, model) -> float:
        """Pricing analítico usando Black-Scholes, Black, o SABR."""
        if not isinstance(model, (BlackScholesModel, BlackModel, SABRModel)):
            raise ValueError("AnalyticalEngine solo funciona con BS, Black, o SABR models")
        
        ttm = product.ttm
        
        if isinstance(model, BlackScholesModel):
            d1 = model.d1(product.S, product.K, ttm)
            d2 = model.d2(product.S, product.K, ttm)
            
            if product.option_type == "call":
                price = (product.S * np.exp(-model.q * ttm) * si.norm.cdf(d1) - 
                        product.K * np.exp(-model.r * ttm) * si.norm.cdf(d2))
            else:
                price = (product.K * np.exp(-model.r * ttm) * si.norm.cdf(-d2) - 
                        product.S * np.exp(-model.q * ttm) * si.norm.cdf(-d1))
        
        elif isinstance(model, BlackModel):
            F = product.F
            d1 = model.d1(F, product.K, ttm)
            d2 = model.d2(F, product.K, ttm)
            discount = np.exp(-model.r * ttm)
            
            if product.option_type == "call":
                price = discount * (F * si.norm.cdf(d1) - product.K * si.norm.cdf(d2))
            else:
                price = discount * (product.K * si.norm.cdf(-d2) - F * si.norm.cdf(-d1))
        
        elif isinstance(model, SABRModel):
            # SABR models use forward prices
            F = product.F if product.F is not None else product.S
            if F is None:
                raise ValueError("SABR models require forward price F. Set product.F or use spot price as forward.")
            
            # Get implied volatility from SABR model
            sigma = model.implied_volatility(F, product.K, ttm)
            
            # Use Black's formula (same as BlackModel)
            d1 = (np.log(F / product.K) + 0.5 * sigma**2 * ttm) / (sigma * np.sqrt(ttm))
            d2 = d1 - sigma * np.sqrt(ttm)
            discount = np.exp(-model.r * ttm)
            
            if product.option_type == "call":
                price = discount * (F * si.norm.cdf(d1) - product.K * si.norm.cdf(d2))
            else:
                price = discount * (product.K * si.norm.cdf(-d2) - F * si.norm.cdf(-d1))
        
        return price * product.qty