"""
BuscaFungi - Feature Engineering Module
Feature engineering completo: ambientales + temporales + interacciones
"""

import numpy as np
import pandas as pd
import requests
import logging
from typing import Optional
from datetime import datetime

from . import config
from .utils import (
    calculate_twi,
    encode_cyclical_feature,
    days_in_season,
    estimate_soil_properties,
    estimate_vegetation_type
)

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Extractor de features ambientales y temporales para modelado de hongos
    """

    def __init__(self):
        self.soil_cache = {}
        self.elevation_cache = {}
        self.vegetation_cache = {}

    def get_soil_properties(self, lat: float, lon: float) -> dict:
        """
        Obtiene propiedades del suelo via SoilGrids API

        Propiedades:
        - pH (acidez del suelo)
        - % arcilla
        - % arena
        - Carbono orgánico

        Fallback: Estimación geográfica si API falla
        """
        cache_key = f"{lat:.3f}_{lon:.3f}"

        if cache_key in self.soil_cache:
            return self.soil_cache[cache_key]

        try:
            url = config.SOILGRIDS_API_URL
            params = {
                'lon': lon,
                'lat': lat,
                'property': ['phh2o', 'clay', 'sand', 'soc'],
                'depth': '0-5cm',
                'value': 'mean'
            }

            response = requests.get(url, params=params, timeout=config.SOILGRIDS_TIMEOUT)
            response.raise_for_status()
            data = response.json()

            soil_data = {}

            if 'properties' in data and 'layers' in data['properties']:
                for layer in data['properties']['layers']:
                    prop_name = layer['name']
                    depths = layer.get('depths', [])

                    if depths:
                        value = depths[0]['values'].get('mean')

                        if value is not None:
                            # Conversiones según SoilGrids spec
                            if prop_name == 'phh2o':
                                soil_data['ph'] = value / 10.0
                            elif prop_name == 'clay':
                                soil_data['clay_percent'] = value / 10.0
                            elif prop_name == 'sand':
                                soil_data['sand_percent'] = value / 10.0
                            elif prop_name == 'soc':
                                soil_data['organic_carbon'] = value / 10.0

            # Si falla API, usar estimación
            if not soil_data:
                soil_data = estimate_soil_properties(lat, lon)

            self.soil_cache[cache_key] = soil_data
            return soil_data

        except Exception as e:
            logger.debug(f"SoilGrids API error: {e}. Usando estimación.")
            soil_data = estimate_soil_properties(lat, lon)
            self.soil_cache[cache_key] = soil_data
            return soil_data

    def get_elevation(self, lat: float, lon: float) -> float:
        """
        Obtiene elevación via Open-Elevation API
        """
        cache_key = f"{lat:.3f}_{lon:.3f}"

        if cache_key in self.elevation_cache:
            return self.elevation_cache[cache_key]

        try:
            url = f"{config.ELEVATION_API_URL}?locations={lat},{lon}"
            response = requests.get(url, timeout=config.ELEVATION_TIMEOUT)
            response.raise_for_status()
            elevation = response.json()['results'][0]['elevation']

            self.elevation_cache[cache_key] = elevation
            return elevation

        except Exception as e:
            logger.debug(f"Elevation API error: {e}. Estimando...")
            # Estimación muy simple
            estimated = max(0, (lat - 36) * 150 + np.random.uniform(-100, 100))
            self.elevation_cache[cache_key] = estimated
            return estimated

    def get_vegetation_type(self, lat: float, lon: float, elevation: float) -> str:
        """
        Estima tipo de vegetación
        """
        cache_key = f"{lat:.3f}_{lon:.3f}"

        if cache_key in self.vegetation_cache:
            return self.vegetation_cache[cache_key]

        veg = estimate_vegetation_type(lat, lon, elevation)
        self.vegetation_cache[cache_key] = veg
        return veg

    def extract_environmental_features(
        self,
        lat: float,
        lon: float,
        species: Optional[str] = None
    ) -> dict:
        """
        Extrae features ambientales para una ubicación

        Returns dict con:
        - Suelo: ph, clay_percent, sand_percent, organic_carbon
        - Topografía: elevation, slope, aspect, twi
        - Vegetación: vegetation_type + one-hot encoding
        - Especies-específicas (si species se proporciona)
        """
        features = {}

        # === SUELO ===
        soil = self.get_soil_properties(lat, lon)
        features.update(soil)

        # === ELEVACIÓN ===
        elevation = self.get_elevation(lat, lon)
        features['elevation'] = elevation

        # === TOPOGRAFÍA (SIMULADA - en producción usar DEM) ===
        # Slope y aspect - idealmente desde DEM, aquí simulado
        slope = np.random.uniform(0, 30)
        aspect = np.random.uniform(0, 360)

        features['slope'] = slope
        features['aspect'] = aspect

        # Aspect como features cíclicos
        aspect_sin, aspect_cos = encode_cyclical_feature(aspect, 360)
        features['aspect_sin'] = aspect_sin
        features['aspect_cos'] = aspect_cos

        # TWI
        twi = calculate_twi(elevation, slope)
        features['twi'] = twi

        # === VEGETACIÓN ===
        vegetation = self.get_vegetation_type(lat, lon, elevation)
        features['vegetation_type'] = vegetation

        # === FEATURES DERIVADAS ===
        # Elevación normalizada (0-1 en rango España)
        features['elevation_normalized'] = elevation / 3500.0  # Pico Mulhacén ~3500m

        # === SPECIES-SPECIFIC FEATURES ===
        if species and species in config.SPECIES_CONFIG:
            sp_config = config.SPECIES_CONFIG[species]

            # Dentro de rango de altitud preferido
            alt_range = sp_config['altitude_range']
            features['in_altitude_range'] = int(alt_range[0] <= elevation <= alt_range[1])

            # Vegetación preferida
            preferred_veg = sp_config['preferred_trees']
            features['preferred_vegetation'] = int(vegetation in preferred_veg)

        return features

    def add_temporal_features(
        self,
        features_dict: dict,
        date: datetime,
        species: Optional[str] = None
    ) -> dict:
        """
        Añade features temporales (estacionalidad, etc.)

        NO incluye features meteorológicas (precipitación, temperatura)
        ya que esas vienen del MeteoDataFetcher

        Parameters:
        -----------
        features_dict : dict
            Features existentes (será modificado in-place)
        date : datetime
            Fecha de la observación/predicción
        species : str, optional
            Especie objetivo

        Returns:
        --------
        dict : features_dict actualizado
        """
        # === DÍA DEL AÑO (cíclico) ===
        day_of_year = date.timetuple().tm_yday
        doy_sin, doy_cos = encode_cyclical_feature(day_of_year, 365)
        features_dict['day_of_year_sin'] = doy_sin
        features_dict['day_of_year_cos'] = doy_cos

        # === MES (cíclico) ===
        month = date.month
        month_sin, month_cos = encode_cyclical_feature(month, 12)
        features_dict['month_sin'] = month_sin
        features_dict['month_cos'] = month_cos

        # === TEMPORADA ===
        features_dict['season_spring'] = int(month in [3, 4, 5])
        features_dict['season_summer'] = int(month in [6, 7, 8])
        features_dict['season_autumn'] = int(month in [9, 10, 11])
        features_dict['season_winter'] = int(month in [12, 1, 2])

        # === SPECIES-SPECIFIC TEMPORAL ===
        if species and species in config.SPECIES_CONFIG:
            sp_config = config.SPECIES_CONFIG[species]

            # En temporada de la especie
            season_months = sp_config['season_months']
            in_season, days_from_start = days_in_season(date, season_months)

            features_dict['in_season'] = int(in_season)
            features_dict['days_from_season_start'] = days_from_start if in_season else -1

        return features_dict

    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crea features de interacción (combinaciones no lineales)

        Interacciones ecológicamente relevantes para hongos:
        - pH × organic_carbon (acidez + nutrientes)
        - elevation × precipitation (altitud + lluvia)
        - temperature × soil_moisture (calor + humedad)
        - slope × precipitation (escorrentía)

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame con features base

        Returns:
        --------
        pd.DataFrame : DataFrame con features de interacción añadidas
        """
        df_int = df.copy()

        # === INTERACCIONES SUELO ===
        if 'ph' in df_int.columns and 'organic_carbon' in df_int.columns:
            df_int['ph_x_organic_carbon'] = df_int['ph'] * df_int['organic_carbon']

        if 'clay_percent' in df_int.columns and 'sand_percent' in df_int.columns:
            # Textura del suelo (clay/sand ratio)
            df_int['clay_sand_ratio'] = (
                df_int['clay_percent'] / (df_int['sand_percent'] + 0.1)
            )

        # === INTERACCIONES TOPOGRÁFICAS ===
        if 'elevation' in df_int.columns and 'slope' in df_int.columns:
            df_int['elevation_x_slope'] = df_int['elevation'] * df_int['slope'] / 1000

        # === INTERACCIONES METEOROLÓGICAS ===
        if 'temp_mean_7d' in df_int.columns and 'precip_sum_7d' in df_int.columns:
            # Índice de humedad-temperatura (crítico para hongos)
            df_int['humid_temp_index_7d'] = (
                df_int['precip_sum_7d'] / (df_int['temp_mean_7d'] + 10)
            )

        if 'temp_mean_15d' in df_int.columns and 'precip_sum_15d' in df_int.columns:
            df_int['humid_temp_index_15d'] = (
                df_int['precip_sum_15d'] / (df_int['temp_mean_15d'] + 10)
            )

        if 'slope' in df_int.columns and 'precip_sum_7d' in df_int.columns:
            # Escorrentía estimada
            df_int['runoff_index'] = df_int['slope'] * df_int['precip_sum_7d'] / 100

        # === INTERACCIONES ESPACIALES-TEMPORALES ===
        if 'elevation' in df_int.columns and 'precip_sum_20d' in df_int.columns:
            df_int['elevation_x_precip'] = (
                df_int['elevation'] * df_int['precip_sum_20d'] / 1000
            )

        if 'twi' in df_int.columns and 'precip_sum_15d' in df_int.columns:
            # Acumulación de agua en terreno
            df_int['water_accumulation_index'] = df_int['twi'] * df_int['precip_sum_15d']

        # === INTERACCIONES VEGETACIÓN ===
        # Vegetación × humedad (ej: pinos con lluvia moderada)
        veg_cols = [col for col in df_int.columns if col.startswith('veg_')]
        if len(veg_cols) > 0 and 'precip_sum_15d' in df_int.columns:
            for veg_col in veg_cols:
                df_int[f'{veg_col}_x_precip'] = (
                    df_int[veg_col] * df_int['precip_sum_15d']
                )

        n_new_features = len(df_int.columns) - len(df.columns)
        logger.info(f"  Creadas {n_new_features} features de interacción")

        return df_int

    def extract_features_for_grid(
        self,
        grid_df: pd.DataFrame,
        species: Optional[str] = None,
        date: Optional[datetime] = None,
        meteo_df: Optional[pd.DataFrame] = None,
        add_interactions: bool = True
    ) -> pd.DataFrame:
        """
        Extrae features completas para un grid

        Parameters:
        -----------
        grid_df : pd.DataFrame
            Grid con columnas ['cell_id', 'lat', 'lon']
        species : str, optional
            Especie objetivo (para features species-specific)
        date : datetime, optional
            Fecha objetivo (para features temporales)
        meteo_df : pd.DataFrame, optional
            DataFrame con features meteorológicas ya calculadas
        add_interactions : bool
            Si True, añade features de interacción

        Returns:
        --------
        pd.DataFrame : Grid con todas las features
        """
        logger.info(f"Extrayendo features para {len(grid_df)} celdas...")

        features_list = []

        for idx, row in grid_df.iterrows():
            if idx % 100 == 0 and idx > 0:
                logger.info(f"  {idx}/{len(grid_df)}...")

            cell_features = {
                'cell_id': row['cell_id'],
                'lat': row['lat'],
                'lon': row['lon']
            }

            # Ambientales
            env_features = self.extract_environmental_features(
                row['lat'],
                row['lon'],
                species=species
            )
            cell_features.update(env_features)

            # Temporales
            if date:
                self.add_temporal_features(cell_features, date, species=species)

            features_list.append(cell_features)

        features_df = pd.DataFrame(features_list)

        # One-hot encoding para vegetación
        if 'vegetation_type' in features_df.columns:
            veg_dummies = pd.get_dummies(
                features_df['vegetation_type'],
                prefix='veg'
            )
            features_df = pd.concat([features_df, veg_dummies], axis=1)

        # Merge con meteo features si existen
        if meteo_df is not None:
            features_df = features_df.merge(
                meteo_df,
                on=['lat', 'lon'],
                how='left'
            )

        # Features de interacción
        if add_interactions:
            features_df = self.create_interaction_features(features_df)

        logger.info(f"✅ {len(features_df.columns)} features totales")

        return features_df
