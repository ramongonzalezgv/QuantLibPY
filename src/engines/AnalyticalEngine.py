from typing import Any
import numpy as np
import scipy.stats as si

from ..models.BlackScholesModel import BlackScholesModel
from ..models.BlackModel import BlackModel
from ..models.SABRModel import SABRModel, NormalSABRModel, LognormalSABRModel
from ..products.Swaption import Swaption

from .PricingEngine import PricingEngine

class AnalyticalEngine(PricingEngine):
    """Motor analítico para fórmulas cerradas (Black-Scholes, Black, SABR)."""
    
    def calculate_price(self, product, model) -> float:
        """
        Calcula el precio total del producto.
        
        Returns:
            float: Precio total (ya incluye notional y quantity)
        """
        if not isinstance(model, (BlackScholesModel, BlackModel, SABRModel)):
            raise ValueError("AnalyticalEngine solo funciona con BS, Black, o SABR models")

        if isinstance(model, BlackScholesModel):
            unit_price = self._price_black_scholes(product, model)
        elif isinstance(model, BlackModel):
            unit_price = self._price_black(product, model)
        elif isinstance(model, SABRModel):
            unit_price = self._price_sabr(product, model)
        
        # Escalar por quantity y notional si el producto los tiene
        total_price = unit_price
        
        if hasattr(product, 'qty'):
            total_price *= product.qty
        
        if hasattr(product, 'notional'):
            total_price *= product.notional
        
        return total_price

    def _price_black_scholes(self, product, model: BlackScholesModel) -> float:
        """
        Price using Black-Scholes formula.
        
        Returns:
            float: Precio por unidad de notional
        """
        ttm = product.ttm
        d1 = model.d1(product.S, product.K, ttm)
        d2 = model.d2(product.S, product.K, ttm)
        
        if product.option_type == "call":
            price = (product.S * np.exp(-model.q * ttm) * si.norm.cdf(d1) - 
                    product.K * np.exp(-model.r * ttm) * si.norm.cdf(d2))
        else:
            price = (product.K * np.exp(-model.r * ttm) * si.norm.cdf(-d2) - 
                    product.S * np.exp(-model.q * ttm) * si.norm.cdf(-d1))
        
        return price
    
    def _price_black(self, product, model: BlackModel) -> float:
        """
        Price using Black's formula (for forwards/futures).
        
        Returns:
            float: Precio por unidad de notional
        """
        ttm = product.ttm
        F = product.F
        
        if F is None:
            raise ValueError("Black model requires forward price F. Set product.F")
        
        d1 = model.d1(F, product.K, ttm)
        d2 = model.d2(F, product.K, ttm)
        discount = np.exp(-model.r * ttm)
        
        # Map option types: payer/receiver (swaptions) or call/put (standard options)
        # Payer swaption = Call on rate (profit when F > K)
        # Receiver swaption = Put on rate (profit when K > F)
        is_call = product.option_type in ["call", "payer"]
        
        if is_call:
            unit_price = discount * (F * si.norm.cdf(d1) - product.K * si.norm.cdf(d2))
        else:
            unit_price = discount * (product.K * si.norm.cdf(-d2) - F * si.norm.cdf(-d1))

        # If product provides an annuity/PVBP (e.g., Swaption), scale the unit price
        annuity = None
        try:
            if hasattr(product, 'annuity') and callable(getattr(product, 'annuity')):
                annuity = product.annuity(model.r)
            elif hasattr(product, 'payment_frequency'):
                # fallback: compute annuity inline for products without annuity() method
                freq = int(product.payment_frequency)
                denom = product.calendar.days_in_year if product.calendar else 365.0
                start_offset = (product.swap_start_date - product.valuation_date).days / denom
                delta = 1.0 / freq
                n_pay = int(round(product.tenor * freq))
                if n_pay > 0:
                    times = start_offset + delta * np.arange(1, n_pay + 1)
                    discounts = np.exp(-model.r * times)
                    annuity = float(np.sum(delta * discounts))
        except Exception:
            annuity = None

        if annuity is not None:
            unit_price = unit_price * annuity

        return unit_price
    
    def _price_sabr(self, product, model: SABRModel) -> float:
        """
        Price using SABR model with implied volatility.
        
        Returns:
            float: Precio por unidad de notional
        """
        ttm = product.ttm
        
        # SABR models use forward prices
        F = product.F if product.F is not None else product.S
        if F is None:
            raise ValueError("SABR models require forward price F. Set product.F or use spot price as forward.")
        
        # Get implied volatility from SABR model
        sigma = model.implied_volatility(F, product.K, ttm)
        discount = np.exp(-model.r * ttm)

        # If beta = 0 (Normal SABR), use Bachelier (normal) formula
        if getattr(model, 'beta', None) == 0 or isinstance(model, NormalSABRModel):
            unit_price = self._price_sabr_normal(product, F, sigma, discount, ttm)
        else:
            # Use Black's formula (log-normal SABR)
            unit_price = self._price_sabr_lognormal(product, F, sigma, discount, ttm)

        # Scale by annuity if available
        annuity = None
        try:
            if hasattr(product, 'annuity') and callable(getattr(product, 'annuity')):
                annuity = product.annuity(model.r)
        except Exception:
            annuity = None

        if annuity is not None:
            unit_price = unit_price * annuity

        return unit_price
    
    def _price_sabr_normal(self, product, F: float, sigma: float, discount: float, ttm: float) -> float:
        """Price using Bachelier (normal) formula for Normal SABR."""
        vol_term = sigma * np.sqrt(ttm)
        if vol_term <= 0:
            # fallback to intrinsic
            is_call = product.option_type in ["call", "payer"]
            payoff = max(F - product.K, 0) if is_call else max(product.K - F, 0)
            return discount * payoff
        
        d = (F - product.K) / vol_term
        pdf = si.norm.pdf(d)
        cdf = si.norm.cdf(d)
        
        is_call = product.option_type in ["call", "payer"]
        
        if is_call:
            price = discount * ((F - product.K) * cdf + vol_term * pdf)
        else:
            # put price
            price = discount * ((product.K - F) * si.norm.cdf(-d) + vol_term * pdf)
        
        return price
    
    def _price_sabr_lognormal(self, product, F: float, sigma: float, discount: float, ttm: float) -> float:
        """Price using Black's formula for Lognormal SABR."""
        d1 = (np.log(F / product.K) + 0.5 * sigma**2 * ttm) / (sigma * np.sqrt(ttm))
        d2 = d1 - sigma * np.sqrt(ttm)
        
        is_call = product.option_type in ["call", "payer"]
        
        if is_call:
            price = discount * (F * si.norm.cdf(d1) - product.K * si.norm.cdf(d2))
        else:
            price = discount * (product.K * si.norm.cdf(-d2) - F * si.norm.cdf(-d1))
        
        return price


