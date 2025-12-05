from abc import ABC, abstractmethod
from typing import Dict, Any

# ============================================================================
# MODELS: They define the stochastic behaviour of the underlying asset
# ============================================================================

class StochasticModel(ABC):
    """Clase base para modelos estocásticos."""
    
    @abstractmethod
    def characteristic_function(self, u: complex, params: Dict) -> complex:
        """Función característica del modelo."""
        pass

    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """
        Método obligatorio para devolver los parámetros de mercado (volatilidad, 
        tasas, etc.) para la clave de caché.
        """
        pass