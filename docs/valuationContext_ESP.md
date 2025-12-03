# OptionValuationContext: Orquestación de Valoración con Cache y Paralelización

## Tabla de contenidos
1. [Introducción](#introducción)
2. [Arquitectura](#arquitectura)
3. [Cómo funciona el Cache](#cómo-funciona-el-cache)
4. [Cómo funciona OptionValuationContext](#cómo-funciona-optionvaluationcontext)
5. [Ejemplos de uso](#ejemplos-de-uso)
6. [Ventajas y casos de uso](#ventajas-y-casos-de-uso)

---

## Introducción

**OptionValuationContext** es una clase que orquesta la valoración de opciones (pricing) combinando tres componentes principales:

- **Engine**: el motor de cálculo (Analytical, FFT, Monte Carlo)
- **Model**: el modelo estocástico (Black-Scholes, Heston, etc.)
- **Product**: el producto (EuropeanOption, etc.)

Además proporciona:
- **Cache LRU thread-safe**: evita recalcular el mismo producto+modelo
- **Logging integrado**: para debugging y auditoría
- **Ejecución paralela**: valoración de múltiples productos en paralelo (threads o procesos)
- **API uniforme**: `value_option()` para un producto, `value_options()` para lotes

---

## Arquitectura

### Componentes clave

#### 1. **_make_cache_key(product, model, extra_kwargs)**

Genera una clave única (SHA256) a partir del producto, modelo y parámetros extras.

```python
def _make_cache_key(product: Any, model: Any, extra_kwargs: Dict) -> str:
    """
    Extrae la representación del producto y modelo:
    - Si product.get_parameters() existe, úsalo
    - Si no, usa vars(product) (atributos)
    - Fallback a repr() si todo falla
    
    Serializa a JSON, hashea a SHA256 para una clave estable.
    """
```

**Ventaja**: dos llamadas con exactamente los mismos parámetros generarán la misma clave.

#### 2. **_LRUCache**

Cache LRU (Least Recently Used) thread-safe basado en `OrderedDict`.

```python
class _LRUCache:
    def __init__(self, maxsize: int = 1024):
        self.maxsize = maxsize        # máximo de entradas
        self._d = OrderedDict()       # diccionario ordenado
        self._lock = threading.Lock() # protección concurrencia

    def get(self, key: str):
        # Busca la clave; si existe, la mueve al final (más reciente)
        # Retorna None si no existe

    def set(self, key: str, value: Any):
        # Añade/actualiza clave
        # Si se supera maxsize, elimina la más antigua (FIFO)
```

**Algoritmo LRU**:
1. Mantiene un diccionario ordenado (FIFO → más reciente al final)
2. Cuando se accede a una clave, se mueve al final (marca como reciente)
3. Cuando se llena, elimina el primero (más antiguo)

**Thread-safety**: usa `threading.Lock()` para proteger lecturas/escrituras concurrentes.

#### 3. **OptionValuationContext**

Orquestador principal que:
- Almacena el engine, logger, cache y config de paralelización
- Expone `value_option()` (single) y `value_options()` (batch)
- Maneja cache hits/misses
- Ejecuta en paralelo si se solicita

---

## Cómo funciona el Cache

### Flujo de una valoración con cache

```
┌─────────────────────────────────────────┐
│ value_option(product, model, **kwargs)  │
└────────────────┬────────────────────────┘
                 │
                 ▼
        ┌────────────────────┐
        │ ¿Cache habilitado? │
        └────┬───────────┬───┘
           NO│           │SÍ
             │           ▼
             │    ┌──────────────────────┐
             │    │ Generar clave SHA256 │
             │    │ (product+model+kw)   │
             │    └──────────┬───────────┘
             │               │
             │               ▼
             │        ┌──────────────────┐
             │        │ Buscar en cache  │
             │        └────┬──────────┬──┘
             │          HIT│          │MISS
             │             │          │
             │             ▼          ▼
             │        ┌─────────┐  ┌─────────────────────┐
             │        │ Retorna │  │ Calcular con engine │
             │        │ valor   │  │ (calculate_price)   │
             │        └─────────┘  └──────────┬──────────┘
             │                                 │
             └─────────────────┬───────────────┘
                               │
                               ▼
                      ┌────────────────────┐
                      │ ¿Guardar en cache? │
                      │ (si key válida)    │
                      └────┬────────┬──────┘
                         SÍ │        │ NO
                           ▼        │
                      ┌─────────┐   │
                      │cache.set│   │
                      │(k, val) │   │
                      └─────────┘   │
                                    │
                      ┌─────────────┘
                      │
                      ▼
                  ┌─────────┐
                  │ Retorna │
                  │ precio  │
                  └─────────┘
```

### Ejemplo de cache hit/miss

```python
# Primera llamada → MISS (calcula)
option = EuropeanOption(S=100, K=100, T=30, option_type='call', qty=1)
bs_model = BlackScholesModel(sigma=0.2, r=0.05, q=0.02)

ctx = OptionValuationContext(analytical_engine, cache_enabled=True)

p1 = ctx.value_option(option, bs_model)  # → MISS
# Genera clave SHA256 del (option, bs_model)
# No está en cache → llama engine.calculate_price()
# Guarda resultado en cache

p2 = ctx.value_option(option, bs_model)  # → HIT
# Genera misma clave (mismo product/model)
# Encuentra en cache → retorna directamente sin calcular
# p1 == p2 (exactamente el mismo valor)

# Tercera llamada con K diferente → MISS
option2 = EuropeanOption(S=100, K=110, T=30, option_type='call', qty=1)
p3 = ctx.value_option(option2, bs_model)  # → MISS
# Genera clave DIFERENTE (K=110 vs K=100)
# No está en cache → calcula nuevamente
# p3 ≠ p1 (precios diferentes)
```

### Limitaciones del cache

1. **Parámetros inmutables**: si modificas `product` o `model` después de crear el contexto, el cache no se invalida automáticamente.
   ```python
   opt = EuropeanOption(S=100, K=100, T=30, option_type='call', qty=1)
   p1 = ctx.value_option(opt, bs_model)  # HIT/MISS depende del histórico
   
   opt.S = 110  # MODIFICACIÓN manual (no recomendado)
   p2 = ctx.value_option(opt, bs_model)  # Puede estar en cache pero con S=100
   ```
   **Solución**: crear nuevos objetos en lugar de modificar.

2. **Tamaño limitado**: cuando se alcanza `cache_maxsize`, se elimina la entrada más antigua.
   ```python
   ctx = OptionValuationContext(..., cache_maxsize=128)
   # Si haces 129 valoraciones distintas, la primera se elimina del cache
   ```

3. **Sin invalidación manual**: no hay método para limpiar el cache explícitamente (se podría añadir si es necesario).

---

## Cómo funciona OptionValuationContext

### value_option(): valoración individual

```python
def value_option(
    self,
    product: Any,
    model: Any,
    *,
    use_cache: Optional[bool] = None,
    log: bool = True,
    **kwargs
) -> float:
```

**Parámetros**:
- `product`: EuropeanOption, etc.
- `model`: BlackScholesModel, HestonModel, etc.
- `use_cache`: override de cache_enabled para esta llamada (None = usar default)
- `log`: si False, no registra en logger (default True)
- `**kwargs`: parámetros extra para engine.calculate_price()

**Lógica**:
1. Determina si usar cache (usa_cache si está especificado, si no usa cache_enabled del contexto)
2. Si cache habilitado:
   - Genera clave SHA256
   - Intenta obtener valor del cache
   - Si hit, retorna (sin calcular)
3. Registra en logger (debug): qué se va a valorar
4. Llama `engine.calculate_price(product, model, **kwargs)`
5. Si cache habilitado y clave válida, guarda resultado
6. Registra precio computado
7. Retorna precio

### value_options(): valoración por lotes

```python
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
```

**Parámetros**:
- `products`: lista/iterable de productos
- `model`: modelo único (mismo para todos)
- `use_cache`: override de cache_enabled
- `parallel`: override de self.parallel (True/False)
- `progress_callback`: función `f(idx, price)` que se llama cuando cada precio está listo
- `**kwargs`: parámetros para engine

**Flujo secuencial** (parallel=False):
```
for i, product in enumerate(products):
    price = value_option(product, model, ...)
    results.append(price)
    if progress_callback:
        progress_callback(i, price)
return results
```

**Flujo paralelo** (parallel=True):
```
Crear ThreadPoolExecutor (o ProcessPoolExecutor si use_process_pool=True)
├─ Submeter tarea para cada (idx, product)
├─ Tarea = value_option(product, model, ...)
├─ Esperar a que se completen (as_completed)
├─ Guardar resultado en orden original (results[idx])
└─ Llamar progress_callback cuando cada tarea termina
return results (en mismo orden que entrada)
```

### Ventajas de paralelización

| Caso | Ganancia |
|------|----------|
| **Analytical Engine** | Mínima (cálculo rápido, overhead > beneficio) |
| **Monte Carlo** | **Alta** (simulaciones largas, CPU-bound) |
| **FFT** | **Media** (transformadas rápidas pero múltiples strikes) |

**Ejemplo**: valorar 100 opciones con Monte Carlo:
- Sin paralelización: ~100 * (tiempo por opción)
- Con 4 workers: ~25 * (tiempo por opción) + overhead = **~4x más rápido**

### Engine switching

Una ventaja clave de usar OptionValuationContext es que puedes cambiar el engine en tiempo de ejecución **sin cambiar el código cliente**:

```python
# Usar el mismo context, cambiar engine
ctx = OptionValuationContext(AnalyticalEngine())
p1 = ctx.value_option(opt, bs_model)  # Analytical

ctx.engine = FFTEngine()
p2 = ctx.value_option(opt, bs_model)  # FFT (mismo contexto)

ctx.engine = MonteCarloEngine(n_paths=100000, seed=42)
p3 = ctx.value_option(opt, bs_model)  # MC (mismo contexto)

# Comparar sin cambiar cliente
print(f"Diff Analytical vs FFT: {abs(p1 - p2):.6e}")
print(f"Diff Analytical vs MC: {abs(p1 - p3):.6e}")
```

---

## Ejemplos de uso

### Ejemplo 1: Cache simple

```python
from src.valuation.context import OptionValuationContext
from src.engines.engines import AnalyticalEngine
from src.products.products import EuropeanOption
from src.models.models import BlackScholesModel

engine = AnalyticalEngine()
ctx = OptionValuationContext(engine, cache_enabled=True, cache_maxsize=128)

opt = EuropeanOption(S=100, K=100, T=30, option_type='call', qty=1)
bs = BlackScholesModel(sigma=0.2, r=0.05, q=0.02)

# Primera llamada: MISS (calcula)
p1 = ctx.value_option(opt, bs)
print(f"Precio 1: {p1:.6f}")

# Segunda llamada: HIT (desde cache, instántaneo)
p2 = ctx.value_option(opt, bs)
print(f"Precio 2: {p2:.6f}")

assert p1 == p2  # exactamente igual
```

### Ejemplo 2: Batch secuencial

```python
import numpy as np

strikes = np.linspace(80, 120, 10)
products = [
    EuropeanOption(S=100, K=K, T=30, option_type='call', qty=1)
    for K in strikes
]

ctx = OptionValuationContext(engine, cache_enabled=False, parallel=False)
prices = ctx.value_options(products, bs)

for K, p in zip(strikes, prices):
    print(f"K={K:.1f}: {p:.6f}")
```

### Ejemplo 3: Batch paralelo con progress callback

```python
def on_progress(idx, price):
    print(f"[✓] Opción {idx} completada: {price:.6f}")

ctx = OptionValuationContext(
    MonteCarloEngine(n_paths=50000, seed=42),
    parallel=True,
    max_workers=4
)

prices = ctx.value_options(
    products,
    bs,
    progress_callback=on_progress
)
```

### Ejemplo 4: Engine switching

```python
ctx = OptionValuationContext(AnalyticalEngine())

# Comparar tres engines en el mismo producto
engines_to_test = [
    ('Analytical', AnalyticalEngine()),
    ('FFT', FFTEngine()),
    ('Monte Carlo', MonteCarloEngine(n_paths=100000, seed=42)),
]

results = {}
for name, eng in engines_to_test:
    ctx.engine = eng
    price = ctx.value_option(opt, bs)
    results[name] = price
    print(f"{name}: {price:.6f}")

# Análisis de diferencias
analytical = results['Analytical']
for name, price in results.items():
    if name != 'Analytical':
        diff = abs(analytical - price)
        rel_diff = 100 * diff / analytical
        print(f"  {name} vs Analytical: {diff:.6e} ({rel_diff:.2f}%)")
```

### Ejemplo 5: Logging para debugging

```python
import logging

# Configurar logging en nivel DEBUG para ver cache hits/misses
logging.basicConfig(level=logging.DEBUG)

ctx = OptionValuationContext(
    engine,
    cache_enabled=True,
    cache_maxsize=10
)

# Cada llamada registrará en el logger
p1 = ctx.value_option(opt1, bs)  # DEBUG: Cache key generation, valuation...
p2 = ctx.value_option(opt1, bs)  # DEBUG: Cache hit for key=...
p3 = ctx.value_option(opt2, bs)  # DEBUG: Cache miss, valuation...
```

---

## Ventajas y casos de uso

### ✅ Ventajas principales

| Ventaja | Descripción |
|---------|-------------|
| **Desacoplamiento** | Separa orchestration (context) de pricing (engine) |
| **Reutilización** | El mismo context funciona con cualquier engine |
| **Caching** | Evita recalcular productos idénticos |
| **Logging** | Auditoría y debugging centralizado |
| **Paralelización** | Batch valuations aceleradas (especialmente MC) |
| **Flexibilidad** | Cambiar engines sin modificar código cliente |
| **Uniformidad** | API única (value_option / value_options) |

### 📊 Casos de uso

#### 1. **Backtesting y análisis de sensibilidad**
```python
# Valorar muchas opciones con diferentes strikes/maturities
strikes = np.linspace(80, 120, 50)
maturities = [7, 30, 90, 180, 365]

all_products = [
    EuropeanOption(S=100, K=K, T=T, option_type='call', qty=1)
    for K in strikes
    for T in maturities
]

ctx = OptionValuationContext(
    MonteCarloEngine(n_paths=10000, seed=42),
    parallel=True,
    max_workers=8,
    cache_enabled=True
)

prices = ctx.value_options(all_products, heston_model)
# → 250 valoraciones, paralelizadas y cacheadas
```

#### 2. **Comparación de modelos**
```python
models_to_compare = {
    'BS': BlackScholesModel(sigma=0.2, r=0.05, q=0.02),
    'Heston': HestonModel(kappa=2.0, theta=0.04, ...),
}

ctx = OptionValuationContext(FFTEngine())

results = {}
for model_name, model in models_to_compare.items():
    prices = ctx.value_options(products, model)
    results[model_name] = prices
```

#### 3. **Calibración de parámetros**
```python
from scipy.optimize import minimize

def objective(params):
    # Crear modelo con parámetros candidatos
    model = HestonModel(*params, r=0.05, q=0.02)
    
    # Valorar con context
    ctx.engine = MonteCarloEngine(n_paths=5000, seed=42)
    model_prices = ctx.value_options(market_products, model)
    
    # Minimizar diferencia con precios de mercado
    error = np.sum((np.array(model_prices) - np.array(market_prices))**2)
    return error

result = minimize(objective, x0=[1.0, 0.04, 0.3, -0.5, 0.04])
```

#### 4. **Risk management (Greeks)**
```python
# Calcular greeks para una cartera de opciones
portfolio_options = [...]

ctx = OptionValuationContext(AnalyticalEngine(), cache_enabled=True)

deltas = [GreeksCalculator.delta(opt, model) for opt in portfolio_options]
gammas = [GreeksCalculator.gamma(opt, model) for opt in portfolio_options]
vegas = [GreeksCalculator.vega(opt, model) for opt in portfolio_options]

# Cache acelera estos cálculos (muchas derivadas numéricas)
```

### 🎯 Cuándo usar el context

**Usar OptionValuationContext cuando**:
- ✅ Necesitas valorar múltiples productos (batch)
- ✅ Quieres comparar engines o modelos
- ✅ Requieres paralelización
- ✅ Necesitas caching para optimizar rendimiento
- ✅ Quieres auditoría/logging centralizado

**Usar engine directamente cuando**:
- ✅ Una única valoración simple
- ✅ Engine ya es muy simple (overhead no vale la pena)
- ✅ No necesitas cache ni paralelización

---

## Conclusión

OptionValuationContext es una herramienta de **orquestación y optimización** que:
1. **Simplifica** la API (value_option / value_options)
2. **Optimiza** rendimiento (cache + paralelización)
3. **Facilita** testing y comparación (engine switching)
4. **Centraliza** logging y auditoria
5. **Escala** desde simples valoraciones hasta análisis complejos

Su arquitectura modular permite que crezcas con el proyecto sin refactorizar código cliente.