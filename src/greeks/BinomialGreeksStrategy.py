from typing import Dict
import copy
import math

from .GreeksStrategy import GreeksStrategy
from ..engines.BinomialEngine import BinomialEngine

class BinomialGreeksStrategy(GreeksStrategy):
    """
    Compute greeks using the BinomialEngine by revaluing the price (finite differences)
    and — when available — by extracting the first/second level node values from the tree
    for more accurate Delta/Gamma. Optionally uses Richardson extrapolation between N and 2N.
    """

    def __init__(self, n_steps: int = 200, use_richardson: bool = True):
        self.engine = BinomialEngine(n_steps=n_steps)
        self.use_richardson = use_richardson

    def _clone_and_bump(self, product, **overrides):
        p = copy.deepcopy(product)
        for k, v in overrides.items():
            setattr(p, k, v)
        return p

    def _tree_levels_values(self, product, model, N):
        """
        Build a CRR tree and return:
          (root_price, level1_values_or_None, level2_values_or_None, u, d)

        The returned root_price and level values are consistent with the engine.calculate_price
        outputs used elsewhere (they include whatever quantity handling product.payoff does).
        """
        import numpy as _np
        from ..models.BlackScholesModel import BlackScholesModel

        if not isinstance(model, BlackScholesModel):
            raise ValueError("Binomial tree node-extraction assumes BlackScholesModel (constant sigma)")

        S = product.S
        T = product.ttm
        r = model.r
        q = model.q
        sigma = model.sigma
        N = max(1, int(N))
        dt = T / N
        u = math.exp(sigma * math.sqrt(dt))
        d = 1.0 / u
        p = (math.exp((r - q) * dt) - d) / (u - d)
        df = math.exp(-r * dt)

        # terminal spots and payoffs
        j = _np.arange(N + 1)
        S_T = S * (u ** (N - j)) * (d ** j)
        values = product.payoff(S_T)  # note: payoff in repo returns qty * payoff

        level1 = None
        level2 = None
        for i in range(N - 1, -1, -1):
            values = df * (p * values[:-1] + (1.0 - p) * values[1:])
            if i == 2:
                level2 = values.copy()
            if i == 1:
                level1 = values.copy()

        root_price = values[0]
        return root_price, level1, level2, u, d

    def _get_tree_levels(self, product, model, N):
        """
        Try to obtain tree levels using the engine API if available; otherwise build them locally.
        Expected engine return: (root_price, level1, level2, u, d)
        """
        # Prefer engine-level API if it exists (keeps engine single-run)
        try:
            # some engines may accept N as a kwarg
            if hasattr(self.engine, "price_and_levels"):
                return self.engine.price_and_levels(product, model, N=N)
        except Exception:
            # ignore and fallback to local computation
            pass

        # fallback: compute locally (same logic as older helper)
        return self._tree_levels_values(product, model, N)

    def _richardson(self, small, big):
        return 2.0 * big - small

    def calculate_greeks(self, product, model) -> Dict[str, float]:
        N = getattr(self.engine, "n_steps", None) or self.engine.n_steps if hasattr(self.engine, "n_steps") else 200

        # Try to obtain node values for N
        try:
            price_N, level1_N, level2_N, u_N, d_N = self._get_tree_levels(product, model, N)
        except Exception:
            # If tree extraction fails (e.g., model mismatch), fall back to FD-only approach
            price_N = self.engine.calculate_price(product, model)
            level1_N = level2_N = u_N = d_N = None

        # Optionally compute at 2N and apply Richardson extrapolation for price/delta/gamma
        price = price_N
        delta = None
        gamma = None

        # Helper to compute delta/gamma from level arrays
        def _delta_gamma_from_levels(S, level1, level2, u, d):
            if level1 is None or len(level1) < 2:
                return None, None
            V_up = level1[0]
            V_down = level1[1]
            S_up = S * u
            S_down = S * d
            delta_tree = (V_up - V_down) / (S_up - S_down)

            gamma_tree = None
            if level2 is not None and len(level2) >= 3:
                V_uu, V_ud, V_dd = level2[0], level2[1], level2[2]
                S_uu = S * u * u
                S_ud = S * u * d
                S_dd = S * d * d
                # slopes on branches
                slope_up = (V_uu - V_ud) / (S_uu - S_ud)
                slope_down = (V_ud - V_dd) / (S_ud - S_dd)
                gamma_tree = (slope_up - slope_down) / ((S_uu - S_dd) / 2.0)
            return delta_tree, gamma_tree

        S0 = product.S
        delta_N, gamma_N = _delta_gamma_from_levels(S0, level1_N, level2_N, u_N, d_N)

        if self.use_richardson:
            # attempt to compute 2N values (cheaper if engine caches the tree)
            try:
                price_2N, level1_2N, level2_2N, u_2N, d_2N = self._get_tree_levels(product, model, int(2 * N))
                delta_2N, gamma_2N = _delta_gamma_from_levels(S0, level1_2N, level2_2N, u_2N, d_2N)
                # Extrapolate if values exist
                if price_2N is not None and price_N is not None:
                    price = self._richardson(price_N, price_2N)
                if delta_N is not None and delta_2N is not None:
                    delta = self._richardson(delta_N, delta_2N)
                else:
                    delta = delta_N
                if gamma_N is not None and gamma_2N is not None:
                    gamma = self._richardson(gamma_N, gamma_2N)
                else:
                    gamma = gamma_N
            except Exception:
                # if 2N fails, fall back to N values
                price = price_N
                delta = delta_N
                gamma = gamma_N
        else:
            price = price_N
            delta = delta_N
            gamma = gamma_N

        # If tree-based delta/gamma not available, fallback to finite differences (re-pricing)
        if delta is None:
            bump_S = max(0.01, 0.001 * product.S)
            prod_up = self._clone_and_bump(product, S=product.S + bump_S)
            prod_dn = self._clone_and_bump(product, S=product.S - bump_S)
            price_up = self.engine.calculate_price(prod_up, model)
            price_dn = self.engine.calculate_price(prod_dn, model)
            delta = (price_up - price_dn) / (2.0 * bump_S)

        if gamma is None:
            bump_S = max(0.01, 0.001 * product.S)
            prod_up = self._clone_and_bump(product, S=product.S + bump_S)
            prod_dn = self._clone_and_bump(product, S=product.S - bump_S)
            price_up = self.engine.calculate_price(prod_up, model)
            price_dn = self.engine.calculate_price(prod_dn, model)
            base_price = price if price is not None else self.engine.calculate_price(product, model)
            gamma = (price_up - 2.0 * base_price + price_dn) / (bump_S ** 2)

        # Theta: forward 1 day (if product.T numeric in days)
        theta = 0.0
        try:
            if hasattr(product, "T") and not isinstance(product.T, str) and product.T is not None:
                dt_days = 1.0
                new_days = max(0.0, product.T - dt_days)
                prod_t = self._clone_and_bump(product, T=new_days)
                price_t = self.engine.calculate_price(prod_t, model)
                # Theta per year (negative if value decays)
                theta = (price_t - (price if price is not None else self.engine.calculate_price(product, model))) / ( - (dt_days / 365.0) )
            else:
                theta = 0.0
        except Exception:
            theta = 0.0

        # Vega (central FD) - bump sigma
        bump_sigma = max(1e-4, 0.01)
        original_sigma = model.sigma
        try:
            model.sigma = original_sigma + bump_sigma
            p_up = self.engine.calculate_price(product, model)
            model.sigma = original_sigma - bump_sigma
            p_dn = self.engine.calculate_price(product, model)
            vega = (p_up - p_dn) / (2.0 * bump_sigma)
        finally:
            model.sigma = original_sigma

        # Rho (central FD) - bump r
        bump_r = 1e-4
        original_r = model.r
        try:
            model.r = original_r + bump_r
            p_r_up = self.engine.calculate_price(product, model)
            model.r = original_r - bump_r
            p_r_dn = self.engine.calculate_price(product, model)
            rho = (p_r_up - p_r_dn) / (2.0 * bump_r)
        finally:
            model.r = original_r

        return {
            "Price": price,
            "Delta": delta,
            "Gamma": gamma,
            "Theta": theta,
            "Vega": vega,
            "Rho": rho
        }