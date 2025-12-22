"""
OptionPricingPY - Option pricing models and utilities
"""

__version__ = "0.7.0"
__author__ = "Ramon Gonzalez"

# 2. Importación directa de Clases/Funciones Clave
#    Esto permite un acceso directo como: 'from mi_libreria import BlackScholesModel'

# Importar desde models
from .models.StochasticModel import StochasticModel
from .models.BlackScholesModel import BlackScholesModel
from .models.BlackModel import BlackModel
from .models.HestonModel import HestonModel

# Importar desde products
from .products.FinancialProduct import FinancialProduct
from .products.EuropeanOption import EuropeanOption
from .products.AsianOption import AsianOption
from .products.AmericanOption import AmericanOption

# Importar desde engines
from .engines.PricingEngine import PricingEngine
from .engines.AnalyticalEngine import AnalyticalEngine
from .engines.FFTEngine import FFTEngine
from .engines.MonteCarloEngine import MonteCarloEngine
from .engines.BinomialEngine import BinomialEngine

from .greeks.GreeksCalculator import GreeksCalculator

# 3. Definición del __all__ principal
#    Esto combina las clases/funciones de todos los subpaquetes
__all__ = [
    "__version__",  # Incluir siempre la versión
    
    # Modelos
    "StochasticModel",
    "BlackScholesModel",
    "BlackModel",
    "HestonModel",
    
    # Productos
    "FinancialProduct",
    "EuropeanOption",
    "AsianOption",
    "AmericanOption",
    
    # Engines
    "PricingEngine",
    "AnalyticalEngine",
    "FFTEngine",
    "MonteCarloEngine",
    "BinomialEngine",

    #Greeks
    "GreeksCalculator"
]