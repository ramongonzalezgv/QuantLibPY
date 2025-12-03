# ...existing code...
from typing import Any, Iterable, List, Optional, Callable, Dict
import logging
import json
import hashlib
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import threading

def _make_cache_key(product: Any, model: Any, extra_kwargs: Dict) -> str:
    """
    Build a stable key from product.get_parameters() (if available) or vars(product),
    and from model.__dict__ (if available). Falls back to repr() for unknowns.
    """
    try:
        prod_repr = product.get_parameters() if hasattr(product, "get_parameters") else vars(product)
    except Exception:
        prod_repr = repr(product)

    try:
        model_repr = vars(model)
    except Exception:
        model_repr = repr(model)

    payload = {"product": prod_repr, "model": model_repr, "kwargs": extra_kwargs}
    serialized = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

class _LRUCache:
    """Tiny thread-safe LRU cache based on OrderedDict."""
    def __init__(self, maxsize: int = 1024):
        self.maxsize = maxsize
        self._d = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str):
        with self._lock:
            try:
                val = self._d.pop(key)
                # move to end (most recently used)
                self._d[key] = val
                return val
            except KeyError:
                return None

    def set(self, key: str, value: Any):
        with self._lock:
            if key in self._d:
                self._d.pop(key)
            self._d[key] = value
            if len(self._d) > self.maxsize:
                # pop oldest
                self._d.popitem(last=False)

class OptionValuationContext:
    """
    Context that orchestrates engine + model + product and provides:
      - logging hook
      - optional result caching (LRU)
      - batch valuation with optional parallel execution
    Assumes engines expose calculate_price(product, model, **kwargs).
    """

    def __init__(
        self,
        engine: Any,
        *,
        logger: Optional[logging.Logger] = None,
        cache_enabled: bool = False,
        cache_maxsize: int = 1024,
        parallel: bool = False,
        max_workers: Optional[int] = None,
        use_process_pool: bool = False,
    ):
        self.engine = engine
        self.logger = logger or logging.getLogger(__name__)
        self.cache_enabled = cache_enabled
        self.cache = _LRUCache(cache_maxsize) if cache_enabled else None
        self.parallel = parallel
        self.max_workers = max_workers
        self.use_process_pool = use_process_pool

    def value_option(self, product: Any, model: Any, *, use_cache: Optional[bool] = None, log: bool = True, **kwargs) -> float:
        """
        Value a single product. kwargs passed to engine.calculate_price.
        use_cache overrides the context default for this call.
        """
        cache_flag = self.cache_enabled if use_cache is None else bool(use_cache)
        key = None
        if cache_flag:
            try:
                key = _make_cache_key(product, model, kwargs)
                cached = self.cache.get(key)
                if cached is not None:
                    if log:
                        self.logger.debug("Cache hit for key=%s", key)
                    return cached
            except Exception as e:
                # If key generation fails, continue without cache but log
                self.logger.debug("Cache key generation failed: %s", e)

        if log:
            self.logger.debug("Valuing product=%s model=%s kwargs=%s", 
                              getattr(product, "get_parameters", lambda: repr(product))(),
                              getattr(model, "__dict__", lambda: repr(model)),
                              kwargs)

        price = self.engine.calculate_price(product, model, **kwargs)

        if cache_flag and key is not None:
            try:
                self.cache.set(key, price)
            except Exception as e:
                self.logger.debug("Failed to set cache: %s", e)

        if log:
            self.logger.debug("Price computed: %s", price)
        return price

    def value_options(
        self,
        products: Iterable[Any],
        model: Any,
        *,
        use_cache: Optional[bool] = None,
        parallel: Optional[bool] = None,
        progress_callback: Optional[Callable[[int, float], None]] = None,
        **kwargs
    ) -> List[float]:
        """
        Value a collection of products. Returns list of prices in the same order.
        - parallel=True will use a ThreadPoolExecutor by default (safer for pickling).
          Set use_process_pool=True in constructor to try ProcessPoolExecutor.
        - progress_callback(index, price) is called when each result is ready.
        """
        prod_list = list(products)
        if not prod_list:
            return []

        do_parallel = self.parallel if parallel is None else bool(parallel)

        if not do_parallel:
            results = []
            for i, p in enumerate(prod_list):
                price = self.value_option(p, model, use_cache=use_cache, **kwargs)
                results.append(price)
                if progress_callback:
                    try:
                        progress_callback(i, price)
                    except Exception:
                        pass
            return results

        # Parallel execution
        Executor = ProcessPoolExecutor if self.use_process_pool else ThreadPoolExecutor
        # If using process pool, engine/product/model must be picklable; try to detect and fallback.
        if self.use_process_pool:
            try:
                import pickle
                pickle.dumps(self.engine)
            except Exception:
                self.logger.warning("Engine not picklable, falling back to ThreadPoolExecutor")
                Executor = ThreadPoolExecutor

        max_workers = self.max_workers or min(32, (len(prod_list) or 1))
        results = [None] * len(prod_list)

        def _task(idx, prod):
            # call underlying engine directly (avoid nested executor recursion)
            return idx, self.value_option(prod, model, use_cache=use_cache, **kwargs)

        with Executor(max_workers=max_workers) as ex:
            futures = {ex.submit(_task, i, p): i for i, p in enumerate(prod_list)}
            for fut in as_completed(futures):
                try:
                    idx, price = fut.result()
                except Exception as e:
                    self.logger.exception("Task failed: %s", e)
                    price = None
                    idx = futures[fut]
                results[idx] = price
                if progress_callback:
                    try:
                        progress_callback(idx, price)
                    except Exception:
                        pass

        return results
