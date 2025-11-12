"""
BuscaFungi - Configuration Module
Configuración centralizada del sistema de predicción de hongos
"""

import numpy as np

# ==================== GRID CONFIGURATION ====================
GRID_RESOLUTION_KM = 1.0  # 1km resolution
USE_SAMPLE = False  # False = full grid, True = sample for testing
SAMPLE_SIZE = 1000  # Samples to use when USE_SAMPLE=True

# ==================== REGION BOUNDARIES ====================
REGIONS = {
    'leon': {'lat_min': 42.2, 'lat_max': 43.0, 'lon_min': -6.5, 'lon_max': -5.0},
    'galicia': {'lat_min': 42.0, 'lat_max': 43.5, 'lon_min': -9.0, 'lon_max': -7.0},
    'pirineos': {'lat_min': 42.3, 'lat_max': 43.0, 'lon_min': -1.5, 'lon_max': 2.0},
    'cataluna': {'lat_min': 40.5, 'lat_max': 42.9, 'lon_min': 0.2, 'lon_max': 3.3},
    'full_spain': {'lat_min': 36.0, 'lat_max': 43.8, 'lon_min': -9.3, 'lon_max': 3.3}
}

FOCUS_REGION = 'full_spain'
SPAIN_BOUNDS = REGIONS[FOCUS_REGION]

# ==================== SPECIES CONFIGURATION ====================
SPECIES_CONFIG = {
    'Boletus edulis': {
        'name': 'Boleto',
        'emoji': '🍄',
        'color': '#8B4513',
        'season_months': [9, 10, 11],  # Sept-Nov (otoño)
        'season_name': 'Otoño',
        'preferred_trees': ['pine', 'oak', 'beech'],
        'altitude_range': (500, 2000),
        'optimal_temp_range': (10, 20),  # °C
        'min_rain_7d': 20,  # mm mínimo en 7 días
        'habitat': 'Bosques de coníferas y caducifolios'
    },
    'Lactarius deliciosus': {
        'name': 'Níscalo',
        'emoji': '🧡',
        'color': '#FF6347',
        'season_months': [10, 11, 12],  # Oct-Dic
        'season_name': 'Otoño-Invierno',
        'preferred_trees': ['pine'],
        'altitude_range': (300, 1800),
        'optimal_temp_range': (8, 18),
        'min_rain_7d': 15,
        'habitat': 'Pinares'
    },
    'Morchella esculenta': {
        'name': 'Colmenilla',
        'emoji': '🌸',
        'color': '#DAA520',
        'season_months': [3, 4, 5],  # Mar-May (primavera)
        'season_name': 'Primavera',
        'preferred_trees': ['ash', 'elm', 'apple'],
        'altitude_range': (200, 1500),
        'optimal_temp_range': (12, 22),
        'min_rain_7d': 25,
        'habitat': 'Bosques de ribera, huertos, zonas quemadas'
    },
}

# ==================== MODEL CONFIGURATION ====================
XGBOOST_PARAMS = {
    'n_estimators': 200,  # Aumentado
    'max_depth': 7,  # Aumentado
    'learning_rate': 0.05,  # Reducido para más control
    'min_child_weight': 3,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'gamma': 0.1,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'scale_pos_weight': 1.5,  # Dar más peso a presencias
    'random_state': 42,
    'n_jobs': -1
}

# ==================== CLUSTERING CONFIGURATION ====================
GMM_N_COMPONENTS = 12  # Aumentado para capturar más nichos
GMM_COVARIANCE_TYPE = 'full'

# ==================== PSEUDO-ABSENCE CONFIGURATION ====================
PSEUDO_ABSENCE_RATIO = 2.0  # 2 ausencias por cada presencia
MIN_DISTANCE_KM = 10  # Distancia mínima a presencias (km)
MAX_ATTEMPTS_RATIO = 3  # Intentos máximos = ratio * n_presencias * MAX_ATTEMPTS_RATIO

# ==================== TEMPORAL CONFIGURATION ====================
TEMPORAL_FEATURES = {
    'precipitation_windows': [7, 15, 20],  # días
    'temperature_windows': [7, 15],  # días
    'sunshine_windows': [7, 15, 20],  # días
    'rain_threshold_mm': 5.0,  # Para calcular "días desde última lluvia"
}

# Slider temporal
SLIDER_PAST_DAYS = 30
SLIDER_FUTURE_DAYS = 15

# ==================== SPATIAL VALIDATION ====================
SPATIAL_BLOCK_SIZE_DEG = 0.25  # ~25km blocks para validación espacial
N_SPATIAL_FOLDS = 5

# ==================== API CONFIGURATION ====================
# Open-Meteo (no requiere API key)
METEO_API_URL = "https://archive-api.open-meteo.com/v1/archive"
METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# SoilGrids
SOILGRIDS_API_URL = "https://rest.isric.org/soilgrids/v2.0/properties/query"
SOILGRIDS_TIMEOUT = 10

# Open-Elevation
ELEVATION_API_URL = "https://api.open-elevation.com/api/v1/lookup"
ELEVATION_TIMEOUT = 3

# ==================== CACHE CONFIGURATION ====================
CACHE_DIR = "data/cache"
ENABLE_CACHE = True
CACHE_EXPIRY_DAYS = 7

# ==================== VISUALIZATION ====================
MAP_DEFAULT_ZOOM = 6
MAP_CENTER_SPAIN = [40.4, -3.7]  # Madrid
MAP_TILES = 'OpenStreetMap'

PROBABILITY_THRESHOLDS = {
    'very_high': 75,
    'high': 60,
    'medium': 40,
    'low': 20
}

COLOR_SCALE = {
    'very_high': 'darkgreen',
    'high': 'green',
    'medium': 'yellow',
    'low': 'orange',
    'very_low': 'lightgray'
}

# ==================== LOGGING ====================
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
