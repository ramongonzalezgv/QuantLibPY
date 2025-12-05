from ..models.BlackScholesModel import BlackScholesModel
from ..models.BlackModel import BlackModel
from ..models.HestonModel import HestonModel
import numpy as np
from scipy.fft import ifft
from scipy.interpolate import interp1d


from .PricingEngine import PricingEngine

class FFTEngine(PricingEngine):
    """Motor FFT usando el método de Lewis."""
    
    def __init__(self, N: int = 2**12, B: float = 200, interp: str = "cubic"):
        self.N = N
        self.B = B
        self.interp = interp
    
    def calculate_price(self, product, model) -> float:
        """Pricing usando FFT y método de Lewis."""
        dx = self.B / self.N
        x = np.arange(self.N) * dx
        
        # Simpson weights
        weight = np.arange(self.N)
        weight = 3 + (-1) ** (weight + 1)
        weight[0] = 1
        weight[self.N - 1] = 1
        
        dk = 2 * np.pi / self.B
        b = self.N * dk / 2
        ks = -b + dk * np.arange(self.N)
        
        # Parámetros para la función característica
        params = {'ttm': product.ttm}
        if isinstance(model, BlackModel):
            params['F'] = product.F
        
        # Integrando
        integrand = (np.exp(-1j * b * np.arange(self.N) * dx) * 
                    model.characteristic_function(x - 0.5j, params) * #type: ignore
                    1 / (x**2 + 0.25) * weight * dx / 3)
        integral_value = np.real(ifft(integrand) * self.N) #type: ignore
        
        # Interpolación
        spline = interp1d(ks, integral_value, kind=self.interp)
        
        # Precio
        if isinstance(model, BlackScholesModel):
            log_moneyness = np.log(product.S / product.K)
            call_price = (product.S * np.exp(-model.q * product.ttm) - 
                         np.sqrt(product.S * product.K) * 
                         np.exp(-model.r * product.ttm) / np.pi * 
                         spline(log_moneyness))
        elif isinstance(model, BlackModel):
            log_moneyness = np.log(product.F / product.K)
            call_price = (np.exp(-model.r * product.ttm) * 
                         (product.F - np.sqrt(product.F * product.K) / 
                          np.pi * spline(log_moneyness)))
        elif isinstance(model, HestonModel):
            log_moneyness = np.log(product.S / product.K)
            call_price = (product.S * np.exp(-model.q * product.ttm) - 
                         np.sqrt(product.S * product.K) * 
                         np.exp(-model.r * product.ttm) / np.pi * 
                         spline(log_moneyness))
        
        # Put-Call Parity si es put
        if product.option_type == "put":
            if isinstance(model, BlackModel):
                call_price += (np.exp(-model.r * product.ttm) * 
                              (product.K - product.F))
            else:
                r = model.r
                q = getattr(model, 'q', 0)
                call_price += (product.K * np.exp(-r * product.ttm) - 
                              product.S * np.exp(-q * product.ttm))
        
        return call_price * product.qty