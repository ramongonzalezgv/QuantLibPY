from typing import Dict
import math

import numpy as np
import scipy.stats as si

from .GreeksStrategy import GreeksStrategy
from ..engines.AnalyticalEngine import AnalyticalEngine
from ..models.BlackScholesModel import BlackScholesModel
from ..models.BlackModel import BlackModel

class AnalyticalGreeksStrategy(GreeksStrategy):
    """Analytical (closed-form) greeks for models that support it (BS / Black)."""

    def calculate_greeks(self, product, model) -> Dict[str, float]:
        # Basic input
        S = product.S
        K = product.K
        t = product.ttm  # in years
        qty = product.qty

        # Price per unit (AnalyticalEngine returns price*qty; divide to get per-unit price)
        price_per_unit = AnalyticalEngine().calculate_price(product, model) / max(qty, 1)

        # Guard for zero time to maturity
        if t <= 0:
            intrinsic = max(S - K, 0) if product.option_type == "call" else max(K - S, 0)
            return {
                "Price": intrinsic * qty,
                "Delta": (1.0 if product.option_type == "call" and S > K else 0.0) * qty,
                "Gamma": 0.0,
                "Theta": 0.0,
                "Vega": 0.0,
                "Rho": 0.0
            }

        # Support Black-Scholes and Black (use model-specific d1/d2 where available)
        if isinstance(model, BlackScholesModel):
            sigma = model.sigma
            r = model.r
            q = model.q
            d1 = model.d1(S, K, t)
            d2 = model.d2(S, K, t)
            pdf_d1 = si.norm.pdf(d1)
            cdf_d1 = si.norm.cdf(d1)
            cdf_d2 = si.norm.cdf(d2)

            # Delta
            if product.option_type == "call":
                delta_unit = math.exp(-q * t) * cdf_d1
            else:
                delta_unit = math.exp(-q * t) * (cdf_d1 - 1.0)

            # Gamma (same for calls/puts)
            gamma_unit = math.exp(-q * t) * pdf_d1 / (S * sigma * math.sqrt(t))

            # Vega (per 1.0 vol change)
            vega_unit = S * math.exp(-q * t) * pdf_d1 * math.sqrt(t)

            # Theta (per year)
            first_term = - (S * pdf_d1 * sigma * math.exp(-q * t)) / (2.0 * math.sqrt(t))
            if product.option_type == "call":
                theta_unit = (first_term
                              - r * K * math.exp(-r * t) * cdf_d2
                              + q * S * math.exp(-q * t) * cdf_d1)
            else:
                theta_unit = (first_term
                              + r * K * math.exp(-r * t) * si.norm.cdf(-d2)
                              - q * S * math.exp(-q * t) * si.norm.cdf(-d1))

            # Rho (per 1.0 change in r)
            if product.option_type == "call":
                rho_unit = K * t * math.exp(-r * t) * cdf_d2
            else:
                rho_unit = -K * t * math.exp(-r * t) * si.norm.cdf(-d2)

            # Package results and multiply by qty
            return {
                "Price": price_per_unit * qty,
                "Delta": delta_unit * qty,
                "Gamma": gamma_unit * qty,
                "Theta": theta_unit * qty,
                "Vega": vega_unit * qty,
                "Rho": rho_unit * qty
            }

        elif isinstance(model, BlackModel):
            # Black model (forward) greeks - F used instead of S
            # Very similar to BS but no q and using forward F and discounting
            F = product.F
            if F is None:
                raise ValueError("BlackModel requires forward price set in product.F")
            sigma = model.sigma
            r = model.r
            d1 = model.d1(F, K, t)
            d2 = model.d2(F, K, t)
            pdf_d1 = si.norm.pdf(d1)
            cdf_d1 = si.norm.cdf(d1)
            cdf_d2 = si.norm.cdf(d2)
            discount = math.exp(-r * t)

            # Delta (w.r.t. underlying spot is model-dependent; here we return option derivative w.r.t. forward)
            if product.option_type == "call":
                delta_unit = discount * cdf_d1
            else:
                delta_unit = discount * (cdf_d1 - 1.0)

            gamma_unit = discount * pdf_d1 / (F * sigma * math.sqrt(t))
            vega_unit = discount * F * pdf_d1 * math.sqrt(t)

            first_term = - (discount * F * pdf_d1 * sigma) / (2.0 * math.sqrt(t))
            if product.option_type == "call":
                theta_unit = (first_term - r * K * discount * cdf_d2)
                rho_unit = K * t * discount * cdf_d2
            else:
                theta_unit = (first_term + r * K * discount * si.norm.cdf(-d2))
                rho_unit = -K * t * discount * si.norm.cdf(-d2)

            return {
                "Price": price_per_unit * qty,
                "Delta": delta_unit * qty,
                "Gamma": gamma_unit * qty,
                "Theta": theta_unit * qty,
                "Vega": vega_unit * qty,
                "Rho": rho_unit * qty
            }

        else:
            raise ValueError("AnalyticalGreeksStrategy only supports BlackScholesModel or BlackModel.")

