from typing import Dict, Optional, Tuple
import json
from functools import lru_cache

from ..products.EuropeanOption import EuropeanOption
from ..products.AmericanOption import AmericanOption
from ..models.BlackScholesModel import BlackScholesModel
from .AnalyticalGreeksStrategy import AnalyticalGreeksStrategy
from .BinomialGreeksStrategy import BinomialGreeksStrategy

def _make_cache_key(product, model, strategy_name: str, opts: dict) -> str:
    """
    Create a stable JSON string key from product.get_parameters(), model.get_parameters(),
    the strategy name and option flags.
    """
    p = {}
    if hasattr(product, "get_parameters"):
        p = product.get_parameters()
    else:
        # best-effort fallback
        p = {k: getattr(product, k) for k in ("S", "K", "T", "option_type", "qty") if hasattr(product, k)}
    m = {}
    if hasattr(model, "get_parameters"):
        m = model.get_parameters()
    else:
        # fallback
        m = {k: getattr(model, k) for k in ("sigma", "r", "q") if hasattr(model, k)}
    key_obj = {
        "product": p,
        "model": m,
        "strategy": strategy_name,
        "opts": opts
    }
    # Ensure determinism
    return json.dumps(key_obj, sort_keys=True, default=str)

class GreeksCalculator:
    """
    Central manager for Greeks strategies.
    - Holds strategy instances (or creates them when needed).
    - Chooses strategy based on product/model and optional overrides.
    - Provides an optional in-memory cache to avoid recomputation.
    """

    def __init__(self,
                 default_n_steps: int = 200,
                 default_use_richardson: bool = True,
                 enable_cache: bool = True):
        self.default_n_steps = default_n_steps
        self.default_use_richardson = default_use_richardson
        self.enable_cache = enable_cache

        # Simple registry of constructors for strategies by name
        self._registry = {
            "analytical": lambda: AnalyticalGreeksStrategy(),
            "binomial": lambda: BinomialGreeksStrategy(n_steps=self.default_n_steps, use_richardson=self.default_use_richardson),
        }

        # Simple in-memory cache: key -> greeks dict
        self._cache = {}

    def register_strategy(self, name: str, constructor):
        """Register a custom strategy constructor (callable returning a GreeksStrategy)."""
        self._registry[name] = constructor

    def _choose_strategy_name(self, product, model, prefer_analytical: bool, force_numerical: bool) -> str:
        """Encapsulate selection logic; override here if you want different defaults."""
        if force_numerical:
            return "binomial"
        if isinstance(product, AmericanOption):
            return "binomial"
        if prefer_analytical and isinstance(model, BlackScholesModel) and isinstance(product, EuropeanOption):
            return "analytical"
        # fallback
        return "binomial"

    def _get_strategy_instance(self, name: str, n_steps: Optional[int], use_richardson: Optional[bool]):
        """
        Instantiate or reconfigure a strategy.
        For strategies that accept parameters, construct them with provided values.
        """
        if name == "binomial":
            n = n_steps if n_steps is not None else self.default_n_steps
            use_r = use_richardson if use_richardson is not None else self.default_use_richardson
            return BinomialGreeksStrategy(n_steps=n, use_richardson=use_r)
        if name == "analytical":
            return AnalyticalGreeksStrategy()
        # custom constructor (callable)
        ctor = self._registry.get(name)
        if ctor is None:
            raise ValueError(f"Unknown strategy: {name}")
        return ctor()

    def clear_cache(self):
        self._cache.clear()

    def all_greeks(self,
                   product,
                   model,
                   *,
                   prefer_analytical: bool = True,
                   force_numerical: bool = False,
                   strategy_name: Optional[str] = None,
                   n_steps: Optional[int] = None,
                   use_richardson: Optional[bool] = None,
                   use_cache: Optional[bool] = None) -> Dict[str, float]:
        """
        Compute all greeks using a selected strategy.

        Parameters:
        - prefer_analytical: prefer closed-form when available.
        - force_numerical: force a numerical strategy (e.g., binomial).
        - strategy_name: explicit strategy name ('analytical'|'binomial'|custom) overrides selection logic.
        - n_steps: overrides binomial steps.
        - use_richardson: override Richardson usage for binomial strategy.
        - use_cache: override the calculator's cache setting for this call.
        """
        use_cache = self.enable_cache if use_cache is None else use_cache

        # Determine strategy
        if strategy_name is None:
            strategy_name = self._choose_strategy_name(product, model, prefer_analytical, force_numerical)

        opts = {
            "n_steps": n_steps or self.default_n_steps,
            "use_richardson": use_richardson if use_richardson is not None else self.default_use_richardson,
            "prefer_analytical": prefer_analytical,
            "force_numerical": force_numerical
        }

        cache_key = _make_cache_key(product, model, strategy_name, opts)
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        # Instantiate strategy
        strategy = self._get_strategy_instance(strategy_name, n_steps=n_steps, use_richardson=use_richardson)

        # Execute
        greeks = strategy.calculate_greeks(product, model)

        # Store in cache
        if use_cache:
            self._cache[cache_key] = greeks

        return greeks