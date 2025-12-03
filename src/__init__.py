"""
OptionPricingPY - Option pricing models and utilities
"""

__version__ = "0.1.0"
__author__ = "Ramon"

# 2. Importación directa de Clases/Funciones Clave
#    Esto permite un acceso directo como: 'from mi_libreria import BlackScholesModel'

# Importar desde models
from .models import (
    StochasticModel,
    BlackScholesModel,
    BlackModel,
    HestonModel,
    GreeksCalculator,
)

# Importar desde products
from .products import (
    FinancialProduct,
    EuropeanOption,
    AsianOption,
)

# Importar desde engines
from .engines import (
    PricingEngine,
    AnalyticalEngine,
    FFTEngine,
    MonteCarloEngine
)

# 3. Definición del __all__ principal
#    Esto combina las clases/funciones de todos los subpaquetes
__all__ = [
    "__version__",  # Incluir siempre la versión
    
    # Modelos
    "StochasticModel",
    "BlackScholesModel",
    "BlackModel",
    "HestonModel",
    "GreeksCalculator",
    
    # Productos
    "FinancialProduct",
    "EuropeanOption",
    "AsianOption",
    
    # Engines
    "PricingEngine",
    "AnalyticalEngine",
    "FFTEngine",
    "MonteCarloEngine",
    
    # Opcional: También puedes exponer los subpaquetes directamente
    "models",
    "products",
    "engines",
]