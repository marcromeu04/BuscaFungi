# 🍄 BuscaFungi

**Sistema Profesional de Predicción Espacio-Temporal de Hongos Comestibles en España**

BuscaFungi es un sistema de machine learning que predice la probabilidad de encontrar hongos comestibles en cualquier ubicación de España, para cualquier fecha.

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
├── models/                    # Modelos entrenados
│
├── train.py                   # Script de entrenamiento
├── predict.py                 # Script de predicción
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

### **Opción 1: Entrenamiento Completo**

```bash
python train.py
```

Esto:
1. Descarga observaciones de GBIF
2. Crea grid de 1km (o muestra si `USE_SAMPLE=True`)
3. Extrae features ambientales
4. Obtiene datos meteorológicos históricos
5. Entrena modelos SDM para cada especie
6. Guarda modelos en `outputs/`

### **Opción 2: Predicción para una fecha**

```python
from src.pipeline import BuscaFungiPipeline
from datetime import datetime

# Cargar pipeline entrenado
pipeline = BuscaFungiPipeline()
pipeline.load_pipeline('outputs/')

# Predecir para hoy
predictions = pipeline.predict_for_date(
    target_date=datetime.now(),
    species='Boletus edulis'
)

# Guardar
predictions.to_csv('predictions_today.csv', index=False)
```

### **Opción 3: Uso como librería**

```python
from src import BuscaFungiPipeline, config

# Configurar
config.FOCUS_REGION = 'galicia'
config.GRID_RESOLUTION_KM = 1.0
config.USE_SAMPLE = False

# Pipeline
pipeline = BuscaFungiPipeline()

# ... (ver ejemplos en notebooks/)
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
