# 🚀 Guía Rápida de Uso - BuscaFungi

## ⚡ Inicio Rápido

### 1. Instalación

```bash
pip install -r requirements.txt
```

### 2. Test Rápido (Recomendado)

```bash
# Test de interpolación meteorológica (100 celdas, ~30 segundos)
python test_interpolation.py
```

**Esperado:**
- ✅ Descarga ~15 puntos de muestra
- ✅ Interpola a 100 celdas
- ✅ Muestra stats de precipitación y temperatura
- ✅ Genera `test_interpolation_output.csv`

### 3. Entrenamiento con Muestra

```bash
# Editar src/config.py primero:
# USE_SAMPLE = True
# SAMPLE_SIZE = 1000

python train.py
```

**Tiempo estimado:** ~5-10 minutos
- Grid: 1000 celdas
- Descarga GBIF: ~2 min
- Features ambientales: ~3 min
- Features meteorológicas: ~2 min
- Entrenamiento: ~1 min

### 4. Entrenamiento Completo

```bash
# Editar src/config.py:
# USE_SAMPLE = False

python train.py
```

**Tiempo estimado:** ~1-2 horas
- Grid: 500,000 celdas
- Features ambientales: ~30 min (con logging cada 50 celdas)
- Features meteorológicas: ~5 min (interpolación)
- Clustering: ~5 min
- Entrenamiento: ~10 min

---

## ⚠️ Problemas Comunes

### Error 400 Bad Request en API Meteorológica

**Causa:** Fecha muy reciente (API archive tiene delay de ~5-7 días)

**Solución:**
```python
# ❌ MAL
target_date = datetime.now() - timedelta(days=2)  # Muy reciente

# ✅ BIEN
target_date = datetime.now() - timedelta(days=30)  # Datos disponibles
```

### Train.py tarda mucho en "estudiar parcelas"

**Causa:** Extracción de features ambientales (API calls)

**Qué ver:**
- Logging cada 50 celdas: `📍 50/1000 celdas (5.0%)`
- Si no ves nada: revisar logging level

**Para ver progress:**
```bash
# Asegurar logging visible
export PYTHONUNBUFFERED=1
python train.py 2>&1 | tee train.log
```

### No sé si está usando interpolación

**Dónde ver:**
- Busca en logs: `🌧️ Obteniendo meteo para N celdas`
- Debe decir: `Puntos de muestreo: ~100` (no 900k)
- Debe decir: `Interpolando a X celdas...`

---

## 📊 Configuración

### `src/config.py` - Opciones clave

```python
# Grid
GRID_RESOLUTION_KM = 1.0      # 1km (cambiar a 0.25 para 250m)
USE_SAMPLE = True              # False para grid completo
SAMPLE_SIZE = 1000             # Celdas si USE_SAMPLE=True

# Región
FOCUS_REGION = 'full_spain'   # 'leon', 'galicia', 'pirineos'

# Modelo
XGBOOST_PARAMS = {...}        # Tuning hiperparámetros

# Temporal (ventanas de agregación)
TEMPORAL_FEATURES = {
    'precipitation_windows': [7, 15, 20],
    'temperature_windows': [7, 15],
    'sunshine_windows': [7, 15, 20]
}

# Pseudo-ausencias
PSEUDO_ABSENCE_RATIO = 2.0    # Ratio ausencias:presencias
MIN_DISTANCE_KM = 10          # Distancia mínima a presencias
```

---

## 🐛 Debug

### Ver todos los logs

```bash
# Cambiar en src/config.py:
LOG_LEVEL = 'DEBUG'  # en vez de 'INFO'
```

### Verificar imports

```python
import sys
sys.path.insert(0, 'src')

from src import BuscaFungiPipeline
from src.meteo import MeteoDataFetcher

print("✅ Imports OK")
```

### Test paso a paso

```python
from src.grid import GridManager
from src.features import FeatureExtractor
from datetime import datetime

# 1. Grid
grid_mgr = GridManager()
grid = grid_mgr.create_grid(use_sample=True, sample_size=100)
print(f"✅ Grid: {len(grid)} celdas")

# 2. Features (sin meteo)
feat_ext = FeatureExtractor()
features = feat_ext.extract_features_for_grid(
    grid,
    date=datetime.now() - timedelta(days=30),
    add_interactions=False
)
print(f"✅ Features: {len(features.columns)} columnas")
```

---

## 💡 Tips

### 1. Empezar pequeño
- USE_SAMPLE=True con SAMPLE_SIZE=100 para probar
- Una vez funcione, aumentar a 1000
- Solo al final usar grid completo

### 2. Usar cache
- Primera ejecución descarga datos
- Segunda ejecución usa cache (instantáneo)
- Cache en: `data/cache/meteo/*.parquet`

### 3. Fechas seguras
- Entrenamiento: observaciones tienen fechas históricas (OK)
- Predicción futura: usar `use_forecast=True`
- Test: usar fecha >= 30 días atrás

### 4. Monitoring
- Logging debe mostrar progreso cada ~30 segundos
- Si no ves nada por >2 minutos en muestra pequeña: problema
- Grid completo puede tardar horas en features ambientales

---

## 📚 Workflow Completo (v2)

### Setup (1 vez)

```bash
# 1. Crear grid + clustering (~25 min)
python setup_grid_clustering.py
```

Genera:
- `outputs/grid_clustered.parquet` - Grid con 15 clusters ecológicos
- `outputs/gmm_model.joblib` - Modelo de clustering
- `outputs/sample_features.parquet` - Features interpoladas

### Entrenamiento

```bash
# 2. Entrenar modelos (~15 min)
python train_v2.py
```

Genera:
- `outputs/models/Boletus_edulis_v2.joblib`
- `outputs/models/Lactarius_deliciosus_v2.joblib`
- `outputs/models/Morchella_esculenta_v2.joblib`
- `outputs/cluster_features.joblib`

### Predicción

```bash
# 3a. Predecir para una fecha histórica
python predict_v2.py --species "Boletus edulis" --date 2024-09-15

# 3b. Predecir para hoy
python predict_v2.py --species "Lactarius deliciosus"

# 3c. Predecir para el futuro (usa forecast)
python predict_v2.py --species "Morchella esculenta" --date 2025-12-01 --use-forecast
```

Genera:
- `outputs/predictions/Boletus_edulis_20240915.csv` - Todas las celdas
- `outputs/predictions/Boletus_edulis_20240915_high_prob.csv` - Solo P > 0.3

### Temporal Slider

```bash
# Predecir para diferentes fechas
for date in 2024-09-{01..30}; do
    python predict_v2.py --species "Boletus edulis" --date $date
done
```

**Ver README.md completo para más detalles**
