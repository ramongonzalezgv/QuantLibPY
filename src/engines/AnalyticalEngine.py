from ..models.BlackScholesModel import BlackScholesModel
from ..models.BlackModel import BlackModel
import numpy as np
import scipy.stats as si


from .PricingEngine import PricingEngine

class AnalyticalEngine(PricingEngine):
    """Motor analítico para fórmulas cerradas (Black-Scholes)."""
    
    def calculate_price(self, product, model) -> float:
        """Pricing analítico usando Black-Scholes."""
        if not isinstance(model, (BlackScholesModel, BlackModel)):
            raise ValueError("AnalyticalEngine solo funciona con BS o Black models")
        
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
        
        return price * product.qty