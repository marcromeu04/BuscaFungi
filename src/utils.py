"""
BuscaFungi - Utility Functions
Funciones auxiliares y helpers
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging

# Setup logger
logger = logging.getLogger(__name__)


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calcula la distancia en kilómetros entre dos puntos en la Tierra
    usando la fórmula de Haversine.

    Parameters:
    -----------
    lat1, lon1 : float
        Latitud y longitud del punto 1 (grados)
    lat2, lon2 : float
        Latitud y longitud del punto 2 (grados)

    Returns:
    --------
    float : Distancia en kilómetros
    """
    # Radio de la Tierra en km
    R = 6371.0

    # Convertir a radianes
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    # Diferencias
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    # Fórmula de Haversine
    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

    distance = R * c
    return distance


def calculate_twi(elevation, slope, catchment_area=100):
    """
    Calcula el Topographic Wetness Index (TWI)

    TWI = ln(a / tan(β))
    donde a = área de captación, β = pendiente

    Valores altos = zonas con acumulación de agua
    """
    slope_rad = np.radians(max(slope, 0.1))  # Evitar división por 0
    twi = np.log((catchment_area + 1) / (np.tan(slope_rad) + 0.001))
    return twi


def days_in_season(date, season_months):
    """
    Calcula si una fecha está en temporada y cuántos días dentro

    Parameters:
    -----------
    date : datetime
        Fecha a evaluar
    season_months : list
        Lista de meses de la temporada [9, 10, 11]

    Returns:
    --------
    tuple : (is_in_season: bool, days_from_start: int)
    """
    month = date.month
    is_in_season = month in season_months

    if not is_in_season:
        return False, 0

    # Calcular días desde inicio de temporada
    season_start = datetime(date.year, min(season_months), 1)
    if date < season_start:
        # Temporada del año anterior
        season_start = datetime(date.year - 1, min(season_months), 1)

    days_from_start = (date - season_start).days
    return True, days_from_start


def encode_cyclical_feature(value, max_value):
    """
    Codifica features cíclicas (día del año, mes, aspecto) en sin/cos

    Parameters:
    -----------
    value : float
        Valor a codificar
    max_value : float
        Valor máximo del ciclo (365 para día del año, 360 para aspecto)

    Returns:
    --------
    tuple : (sin_value, cos_value)
    """
    angle = 2 * np.pi * value / max_value
    return np.sin(angle), np.cos(angle)


def create_spatial_groups(df, lat_col='lat', lon_col='lon', block_size=0.25):
    """
    Crea grupos espaciales para validación cruzada espacial

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con coordenadas
    lat_col, lon_col : str
        Nombres de columnas de latitud/longitud
    block_size : float
        Tamaño del bloque en grados (0.25° ≈ 25km)

    Returns:
    --------
    pd.Series : Grupos espaciales
    """
    spatial_groups = (
        (df[lat_col] // block_size).astype(str) + '_' +
        (df[lon_col] // block_size).astype(str)
    )
    return spatial_groups


def estimate_soil_properties(lat, lon):
    """
    Estimación rápida de propiedades del suelo basada en geografía
    Usado como fallback cuando la API falla

    NOTA: Estas son estimaciones muy simplificadas
    """
    # Norte húmedo
    if lat > 42:
        ph = np.random.uniform(5.5, 6.5)
        organic_carbon = np.random.uniform(30, 50)
        clay_percent = np.random.uniform(20, 35)
        sand_percent = np.random.uniform(25, 45)
    # Sur seco
    elif lat < 38:
        ph = np.random.uniform(7.0, 8.0)
        organic_carbon = np.random.uniform(10, 25)
        clay_percent = np.random.uniform(15, 25)
        sand_percent = np.random.uniform(40, 60)
    # Centro
    else:
        ph = np.random.uniform(6.5, 7.5)
        organic_carbon = np.random.uniform(15, 35)
        clay_percent = np.random.uniform(18, 30)
        sand_percent = np.random.uniform(30, 55)

    return {
        'ph': ph,
        'clay_percent': clay_percent,
        'sand_percent': sand_percent,
        'organic_carbon': organic_carbon
    }


def estimate_vegetation_type(lat, lon, elevation):
    """
    Estimación de tipo de vegetación basada en ubicación y elevación

    SIMPLIFICACIÓN - En producción usar CORINE Land Cover
    """
    # Galicia/Norte atlántico
    if lat > 42 and lon < -2:
        if elevation < 400:
            return 'oak'
        elif elevation < 900:
            return 'beech'
        elif elevation < 1600:
            return 'pine'
        else:
            return 'alpine'
    # Sur mediterráneo
    elif lat < 38:
        if elevation < 400:
            return 'scrubland'
        elif elevation < 1000:
            return 'cork_oak'
        else:
            return 'pine'
    # Resto de España
    else:
        if elevation < 800:
            return 'oak'
        elif elevation < 1500:
            return 'pine'
        else:
            return 'alpine'


def get_season_from_date(date):
    """
    Obtiene la estación del año para una fecha

    Returns: 'spring', 'summer', 'autumn', 'winter'
    """
    month = date.month
    if month in [3, 4, 5]:
        return 'spring'
    elif month in [6, 7, 8]:
        return 'summer'
    elif month in [9, 10, 11]:
        return 'autumn'
    else:
        return 'winter'


def normalize_probability(probs, min_val=0, max_val=100):
    """
    Normaliza probabilidades a rango [min_val, max_val]
    """
    probs = np.array(probs)
    if probs.max() == probs.min():
        return np.full_like(probs, (min_val + max_val) / 2)

    normalized = (probs - probs.min()) / (probs.max() - probs.min())
    scaled = normalized * (max_val - min_val) + min_val
    return scaled


def format_date_range(start_date, end_date):
    """
    Formatea rango de fechas para display
    """
    return f"{start_date.strftime('%Y-%m-%d')} → {end_date.strftime('%Y-%m-%d')}"


def calculate_grid_statistics(df, value_col='probability'):
    """
    Calcula estadísticas descriptivas de una columna
    """
    stats = {
        'mean': df[value_col].mean(),
        'std': df[value_col].std(),
        'min': df[value_col].min(),
        'max': df[value_col].max(),
        'q25': df[value_col].quantile(0.25),
        'q50': df[value_col].quantile(0.50),
        'q75': df[value_col].quantile(0.75),
    }
    return stats


def print_section_header(title, emoji=''):
    """
    Imprime header de sección formateado
    """
    print("\n" + "="*60)
    print(f"{emoji} {title}")
    print("="*60)


def print_progress(current, total, prefix='Progress'):
    """
    Imprime barra de progreso simple
    """
    percent = 100 * (current / total)
    bar_length = 40
    filled = int(bar_length * current / total)
    bar = '█' * filled + '░' * (bar_length - filled)
    print(f'\r{prefix}: |{bar}| {percent:.1f}% ({current}/{total})', end='')
    if current == total:
        print()  # Nueva línea al terminar
