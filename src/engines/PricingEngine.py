from abc import ABC, abstractmethod

# ============================================================================
# ENGINES:Implements numeric and analytical methods for pricing
# ============================================================================

class PricingEngine(ABC):
    """Clase base para motores de valoración."""
    
    @abstractmethod
    def calculate_price(self, product, model) -> float:
        """Calcula el precio del producto."""
        pass
