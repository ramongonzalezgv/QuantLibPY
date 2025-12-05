import numpy as np

from .PricingEngine import PricingEngine
from ..models.BlackScholesModel import BlackScholesModel

class BinomialEngine(PricingEngine):
    """
    Motor de Árbol Binomial (Cox-Ross-Rubinstein).
    Ideal para entender la evolución del precio y adaptable a opciones Americanas.
    """

    def __init__(self, n_steps: int = 1000):
        """
        Parameters
        ----------
        n_steps : int
            Número de pasos en el árbol. Mayor N = mayor precisión, pero más lento.
        """
        self.n_steps = n_steps
        self._cache = {}  # simple in-memory cache; key -> (price, level1, level2, u, d)

    def _cache_key(self, product, model, N):
        # product.get_parameters() and model.get_parameters() should return JSON-serializable dicts
        prod_key = tuple(sorted(product.get_parameters().items()))
        model_key = tuple(sorted(model.get_parameters().items()))
        return (prod_key, model_key, int(N))

    def price_and_levels(self, product, model, levels=(1,2), N=None):
        """
        Compute CRR tree once and return root price and requested early-level node option values.
        Returns: (price, level1_or_None, level2_or_None, u, d)
        """
        if N is None:
            N = self.n_steps
        key = self._cache_key(product, model, N)
        cached = self._cache.get(key)
        if cached is not None:
            # cached is (price, level1, level2, u, d)
            return cached

        # --- same logic as calculate_price but keep intermediate values ---
        S = product.S
        K = product.K
        T = product.ttm
        r = model.r
        q = getattr(model, "q", 0.0)
        sigma = model.sigma
        N = max(1, int(N))
        dt = T / N
        u = np.exp(sigma * np.sqrt(dt))
        d = 1.0 / u
        p = (np.exp((r - q) * dt) - d) / (u - d)
        df = np.exp(-r * dt)

        j = np.arange(N + 1)
        S_T = S * (u ** (N - j)) * (d ** j)
        values = product.payoff(S_T)

        level1 = None
        level2 = None
        for i in range(N - 1, -1, -1):
            values = df * (p * values[:-1] + (1.0 - p) * values[1:])
            if i == 2:
                level2 = values.copy()
            if i == 1:
                level1 = values.copy()

        root_price = values[0]
        result = (root_price, level1, level2, u, d)
        self._cache[key] = result
        return result

    def calculate_price(self, product, model) -> float:
        price, _, _, _, _ = self.price_and_levels(product, model, N=self.n_steps)
        return price * product.qty if product.payoff(np.array([product.S])) is None else price
        