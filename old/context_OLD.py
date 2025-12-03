from typing import Any, Iterable, List

class OptionValuationContext:
    """
    Orquesta engine + model + product.
    - Se asume que el engine expone calculate_price(product, model, **kwargs)
    - product debe exponer la API que ya usas (e.g. EuropeanOption).
    """

    def __init__(self, engine: Any):
        self.engine = engine

    def value_option(self, product: Any, model: Any, **kwargs) -> float:
        """Valora un único producto delegando en el engine."""
        return self.engine.calculate_price(product, model, **kwargs)

    def value_options(self, products: Iterable[Any], model: Any, **kwargs) -> List[float]:
        """Valora una colección de productos y devuelve una lista de precios."""
        return [self.value_option(p, model, **kwargs) for p in products]