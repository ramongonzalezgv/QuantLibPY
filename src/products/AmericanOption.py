from .EuropeanOption import EuropeanOption

class AmericanOption(EuropeanOption):
    """
    American Option.
    Inherits from EuropeanOption because it has the same parameters (S, K, T...),
    but the pricing engine will know that it can be exercised before expiry.
    """

    def info(self) -> str:
        """Información del producto adaptada."""
        base_info = super().info()
        return base_info.replace("European", "American")

    # No need to redefine payoff(), __init__ or ttm
    # because the base logic is the same