# 🍄 BuscaFungi

**Sistema Profesional de Predicción Espacio-Temporal de Hongos Comestibles en España**

BuscaFungi es un sistema de machine learning que predice la probabilidad de encontrar hongos comestibles en cualquier ubicación de España, para cualquier fecha.

> ⚠️ **IMPORTANTE**: Para usar datos meteorológicos históricos, necesitas una API key gratuita de Open-Meteo.
> 📖 **Ver [API_KEY_SETUP.md](API_KEY_SETUP.md)** para instrucciones de configuración (2 minutos)

## 🎯 ¿Qué hace?

- **Predicción espacial**: Mapa de probabilidades de presencia de hongos en grid de 1km
- **Predicción temporal**: Slider de -30 días a +15 días para ver la evolución diaria
- **Features meteorológicas**: Precipitación, temperatura, radiación solar (7d, 15d, 20d)
- **Features ambientales**: Suelo (pH, carbono orgánico), elevación, vegetación, topografía
- **Clustering ecológico**: Identifica zonas ambientalmente similares
- **3 especies**: Boletus edulis, Lactarius deliciosus, Morchella esculenta

---

## 🚀 Características Principales

### ✅ **NUEVAS MEJORAS (v2.0)**

1. **Grid Fijo Determinístico** → Elimina data leakage
2. **Features Temporales Completas**:
   - Precipitación acumulada: 7d, 15d, 20d
   - Temperatura media: 7d, 15d
   - Horas de sol: 7d, 15d, 20d
   - Días desde última lluvia
3. **Integración Open-Meteo API**: Datos históricos + forecast
4. **🚀 Interpolación Espacial Inteligente**:
   - 900k celdas en **~5 minutos** (vs 750 horas sin optimización)
   - Sampling cada 50km + interpolación vectorizada
   - Cache en disco para reutilización
5. **Pseudo-ausencias inteligentes**: Espaciales (>10km) + ecológicas (clusters diferentes)
6. **Feature engineering avanzado**: Interacciones ecológicamente relevantes
7. **Validación espacial**: GroupKFold por bloques de 25km
8. **Código modular**: Arquitectura limpia y profesional
9. **Sin data leakage**: Train/test completamente separados

---

## 📂 Estructura del Proyecto

```
BuscaFungi/
├── src/
│   ├── __init__.py
│   ├── config.py              # Configuración centralizada
│   ├── grid.py                # Grid fijo deterministico
│   ├── meteo.py               # Open-Meteo API integration
│   ├── features.py            # Feature engineering
│   ├── pseudo_absences.py     # Generación inteligente
│   ├── sdm.py                 # Species Distribution Model
│   ├── clustering.py          # GMM clustering
│   ├── pipeline.py            # Pipeline principal
│   └── utils.py               # Funciones auxiliares
│
├── data/
│   ├── raw/                   # Datos crudos
│   ├── processed/             # Datos procesados
│   └── cache/                 # Cache de APIs
│
├── outputs/                   # Modelos y predicciones
├── archive/                   # Código legacy (notebooks viejos)
│
├── setup_grid_clustering.py   # Paso 1: Preprocesamiento del grid
├── train_v2.py                # Paso 2: Entrenamiento de modelos
├── predict_v2.py              # Paso 3: Predicciones
├── test_interpolation.py      # Test de interpolación
├── requirements.txt           # Dependencias
└── README.md                  # Este archivo
```

---

## 🛠️ Instalación

### 1. Clonar repositorio

```bash
git clone https://github.com/marcromeu04/BuscaFungi.git
cd BuscaFungi
```

### 2. Crear entorno virtual (recomendado)

```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# o
venv\Scripts\activate  # Windows
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

---

## 🎓 Uso

### **Workflow v2.0 (3 pasos)**

#### **Paso 1: Preprocesar Grid (ejecutar una vez)**

```bash
python setup_grid_clustering.py
```

Esto:
- Crea grid de 1km para toda España (~900k celdas)
- Extrae features ambientales (suelo, elevación, topografía)
- Obtiene datos meteorológicos con interpolación espacial optimizada
- Realiza clustering ecológico (GMM, 15 componentes)
- Guarda: `outputs/grid_clustered.parquet`, `outputs/gmm_model.joblib`
- Tiempo: ~30 minutos (se ejecuta solo una vez)

#### **Paso 2: Entrenar Modelos**

```bash
python train_v2.py
```

Esto:
- Descarga observaciones de GBIF para las 3 especies
- Carga el grid preprocesado
- Añade features meteorológicas temporales (30d windows)
- Genera pseudo-ausencias inteligentes (espaciales + ecológicas)
- Entrena modelos XGBoost con validación espacial
- Guarda modelos en: `outputs/models/*_v2.joblib`
- Tiempo: ~15 minutos

#### **Paso 3: Hacer Predicciones**

```bash
# Predicción para fecha específica (ej: ayer)
python predict_v2.py --species "Boletus edulis" --date 2024-11-18

# Usar forecast meteorológico (próximos 7 días)
python predict_v2.py --species "Lactarius deliciosus" --date 2024-11-25 --use-forecast

# Todas las especies para hoy
python predict_v2.py
```

Esto:
- Carga grid + modelos entrenados
- Obtiene datos meteorológicos para la fecha objetivo
- Predice probabilidades para todas las celdas
- Guarda: `outputs/predictions/{species}_{date}.csv`
- Tiempo: ~5 minutos

### **Uso Programático (Python)**

```python
from src.grid import create_full_grid
from src.meteo import add_meteorological_features
from src.sdm import train_sdm_model
import joblib
from datetime import datetime

# Cargar grid preprocesado
grid = joblib.load('outputs/grid_clustered.parquet')

# Cargar modelo
model = joblib.load('outputs/models/Boletus_edulis_v2.joblib')

# Añadir meteo para fecha específica
grid_with_meteo = add_meteorological_features(
    grid,
    target_date=datetime(2024, 11, 18),
    use_forecast=False
)

# Predecir
predictions = model.predict_proba(grid_with_meteo)[:, 1]
grid_with_meteo['probability'] = predictions

# Filtrar zonas con alta probabilidad (>60%)
hotspots = grid_with_meteo[grid_with_meteo['probability'] > 0.6]
print(f"Zonas prometedoras: {len(hotspots)} celdas")
```

---

## 📊 Configuración

Edita `src/config.py` para personalizar:

```python
# Grid
GRID_RESOLUTION_KM = 1.0  # Resolución del grid (km)
USE_SAMPLE = False         # True = muestra rápida para testing

# Región
FOCUS_REGION = 'full_spain'  # 'leon', 'galicia', 'pirineos', 'full_spain'

# Modelo
XGBOOST_PARAMS = {...}     # Hiperparámetros XGBoost
GMM_N_COMPONENTS = 12      # Número de clusters ecológicos

# Pseudo-ausencias
PSEUDO_ABSENCE_RATIO = 2.0    # Ratio ausencias:presencias
MIN_DISTANCE_KM = 10          # Distancia mínima a presencias

# Temporal
TEMPORAL_FEATURES = {
    'precipitation_windows': [7, 15, 20],
    'temperature_windows': [7, 15],
    'sunshine_windows': [7, 15, 20]
}
```

---

## 🔬 Metodología

### **1. Adquisición de Datos**

- **Observaciones**: GBIF (Global Biodiversity Information Facility)
- **Suelo**: SoilGrids API
- **Elevación**: Open-Elevation API
- **Meteorología**: Open-Meteo API (histórica + forecast)
  - **Optimización**: Interpolación espacial inteligente
  - Sampling: 1 request cada 50km (~100 puntos para España)
  - Interpolación: Linear + Nearest Neighbor (scipy)
  - Cache: Disco + memoria (parquet)
  - Rendimiento: 900k celdas en ~5 minutos ⚡
- **Vegetación**: Estimación heurística (TODO: CORINE Land Cover)

### **2. Feature Engineering**

#### **Ambientales** (estáticas)
- Suelo: pH, % arcilla, % arena, carbono orgánico
- Topografía: Elevación, pendiente, aspecto, TWI
- Vegetación: Tipo + one-hot encoding

#### **Temporales** (dinámicas)
- Precipitación: suma 7d/15d/20d, máxima, días con lluvia
- Temperatura: media 7d/15d, mínima, máxima
- Radiación solar: horas de sol 7d/15d/20d
- Estacionalidad: día del año, mes, en temporada

#### **Interacciones**
- pH × carbono orgánico
- Elevación × precipitación
- Temperatura × humedad
- Vegetación × precipitación
- (15+ interacciones ecológicas)

### **3. Pseudo-Ausencias Inteligentes**

Estrategia dual:
1. **Espacial**: >10km de cualquier presencia
2. **Ecológica**: Preferencia por clusters ambientales diferentes

Ratio: 2 ausencias por cada presencia

### **4. Modelo**

- **Algoritmo**: XGBoost (Gradient Boosting)
- **Validación**: GroupKFold espacial (5 folds, bloques de 25km)
- **Métricas**: AUC-ROC
- **Calibración**: StandardScaler

### **5. Clustering**

- **Algoritmo**: Gaussian Mixture Model (GMM)
- **N componentes**: 12
- **Features**: Ambientales estáticos (sin temporales)
- **Uso**:
  - Generación de pseudo-ausencias
  - Identificación de nichos ecológicos
  - Visualización de zonas similares

---

## 📈 Resultados Esperados

### **Métricas de Validación**

- **AUC-ROC**: ~0.70-0.85 (validación espacial)
- **Precisión**: Dependiente del threshold (ej: >50% probabilidad)

### **Features Más Importantes (típicamente)**

1. Precipitación acumulada (15-20d)
2. Tipo de vegetación (pino, roble, haya)
3. Elevación
4. Carbono orgánico del suelo
5. Temperatura media (7-15d)
6. pH del suelo
7. Días desde última lluvia
8. Estacionalidad (mes, día del año)

---

## 🐛 Problemas Conocidos y TODOs

### **Limitaciones Actuales**

1. **Vegetación**: Estimación heurística → Usar CORINE Land Cover
2. **Topografía**: Slope/aspect simulados → Usar DEM real
3. **Resolución**: 1km es un compromiso (ideal: 250m, pero 15M celdas)
4. **Dataset incompleto**: Sesgo de muestreo en GBIF
5. **Temporal granularity**: Modelo diario, pero observaciones sin hora exacta

### **TODOs / Mejoras Futuras**

- [ ] Integrar CORINE Land Cover para vegetación real
- [ ] Usar DEM (Digital Elevation Model) para topografía precisa
- [ ] Añadir más especies (100+ hongos comestibles de España)
- [ ] API REST con FastAPI para servir predicciones
- [ ] Frontend interactivo (mapa + slider temporal)
- [ ] Modelo ensemble (XGBoost + Random Forest + Neural Network)
- [ ] Hyperparameter tuning con Optuna
- [ ] MLflow para tracking de experimentos
- [ ] Docker para reproducibilidad
- [ ] CI/CD con GitHub Actions
- [ ] Tests unitarios y de integración

---

## 🤝 Contribuciones

¡Contribuciones son bienvenidas!

1. Fork el proyecto
2. Crea una rama (`git checkout -b feature/amazing-feature`)
3. Commit tus cambios (`git commit -m 'Add amazing feature'`)
4. Push a la rama (`git push origin feature/amazing-feature`)
5. Abre un Pull Request

---

## 📄 Licencia

Este proyecto es de código abierto y está disponible bajo la licencia MIT.

---

## 🙏 Agradecimientos

- **GBIF** por los datos de observaciones
- **Open-Meteo** por datos meteorológicos gratuitos y de calidad
- **SoilGrids** por datos de suelo
- **Comunidad de micólogos** por compartir conocimiento ecológico

---

## 📧 Contacto

- **Autor**: Marc Romeu
- **GitHub**: [@marcromeu04](https://github.com/marcromeu04)
- **Proyecto**: [BuscaFungi](https://github.com/marcromeu04/BuscaFungi)

---

## 📚 Referencias

1. Elith, J., & Leathwick, J. R. (2009). Species distribution models: ecological explanation and prediction across space and time. *Annual Review of Ecology, Evolution, and Systematics*, 40, 677-697.

2. Valavi, R., Elith, J., Lahoz-Monfort, J. J., & Guillera-Arroita, G. (2019). blockCV: An R package for generating spatially or environmentally separated folds for k-fold cross-validation of species distribution models. *Methods in Ecology and Evolution*, 10(2), 225-232.

3. Barbet-Massin, M., Jiguet, F., Albert, C. H., & Thuiller, W. (2012). Selecting pseudo-absences for species distribution models: how, where and how many?. *Methods in Ecology and Evolution*, 3(2), 327-338.

4. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. In *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 785-794).

---

**🍄 ¡Feliz búsqueda de hongos! 🍄**
