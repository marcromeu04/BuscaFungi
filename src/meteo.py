"""
BuscaFungi - Meteorological Data Module
Integración con Open-Meteo API para datos históricos y forecast
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional

from . import config

logger = logging.getLogger(__name__)


class MeteoDataFetcher:
    """
    Cliente para Open-Meteo API

    Obtiene:
    - Datos históricos (archive): 1940-presente
    - Datos actuales: últimos 90 días
    - Forecast: próximos 16 días

    OPTIMIZACIÓN:
    - Cache en memoria y disco
    - Interpolación espacial para grids grandes
    """

    def __init__(self, enable_disk_cache: bool = True, api_key: str = None):
        self.archive_url = config.METEO_API_URL
        self.forecast_url = config.METEO_FORECAST_URL
        self.cache = {}  # Cache en memoria
        self.api_key = api_key or config.METEO_API_KEY  # From config or env

        # Cache en disco
        self.enable_disk_cache = enable_disk_cache
        if enable_disk_cache:
            from pathlib import Path
            self.cache_dir = Path(config.CACHE_DIR) / 'meteo'
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_key(self, lat: float, lon: float, start_date: datetime, end_date: datetime) -> str:
        """Genera cache key único para una request"""
        return f"{lat:.4f}_{lon:.4f}_{start_date.date()}_{end_date.date()}"

    def _load_from_disk_cache(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Carga datos desde caché de disco"""
        if not self.enable_disk_cache:
            return None

        cache_file = self.cache_dir / f"{cache_key}.parquet"

        if cache_file.exists():
            try:
                df = pd.read_parquet(cache_file)
                logger.debug(f"Cache hit (disco): {cache_key}")
                return df
            except Exception as e:
                logger.debug(f"Error leyendo cache: {e}")
                return None

        return None

    def _save_to_disk_cache(self, cache_key: str, df: pd.DataFrame):
        """Guarda datos en caché de disco"""
        if not self.enable_disk_cache:
            return

        cache_file = self.cache_dir / f"{cache_key}.parquet"

        try:
            df.to_parquet(cache_file)
            logger.debug(f"Cache guardado: {cache_key}")
        except Exception as e:
            logger.debug(f"Error guardando cache: {e}")

    def fetch_historical_weather(
        self,
        lat: float,
        lon: float,
        start_date: datetime,
        end_date: datetime,
        variables: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Obtiene datos meteorológicos históricos para una ubicación

        ⚠️ IMPORTANTE: API archive tiene delay de ~5-7 días
        No usar fechas muy recientes (usar end_date <= hoy - 7 días)

        Parameters:
        -----------
        lat, lon : float
            Coordenadas
        start_date, end_date : datetime
            Rango de fechas
        variables : list, optional
            Variables a obtener. Default: todas las necesarias

        Returns:
        --------
        pd.DataFrame : Datos meteorológicos diarios
        """
        if variables is None:
            variables = [
                'temperature_2m_mean',
                'temperature_2m_max',
                'temperature_2m_min',
                'precipitation_sum',
                'rain_sum',
                'snowfall_sum',
                'sunshine_duration',
                'soil_moisture_0_to_7cm'
            ]

        # Cache en memoria
        cache_key = self._get_cache_key(lat, lon, start_date, end_date)
        if cache_key in self.cache:
            logger.debug(f"Cache hit (memoria): {cache_key}")
            return self.cache[cache_key]

        # Cache en disco
        cached_df = self._load_from_disk_cache(cache_key)
        if cached_df is not None:
            self.cache[cache_key] = cached_df
            return cached_df

        # Retry con exponential backoff
        import time
        max_retries = 3

        params = {
            'latitude': lat,
            'longitude': lon,
            'start_date': start_date.strftime('%Y-%m-%d'),
            'end_date': end_date.strftime('%Y-%m-%d'),
            'daily': ','.join(variables),
            'timezone': 'Europe/Madrid'
        }

        # Añadir API key si está disponible
        if self.api_key:
            params['apikey'] = self.api_key

        for attempt in range(max_retries):
            try:
                response = requests.get(self.archive_url, params=params, timeout=60)
                response.raise_for_status()
                data = response.json()

                # Convertir a DataFrame
                df = pd.DataFrame(data['daily'])
                df['time'] = pd.to_datetime(df['time'])
                df = df.rename(columns={'time': 'date'})

                # Cache en memoria y disco
                self.cache[cache_key] = df
                self._save_to_disk_cache(cache_key, df)

                return df

            except requests.exceptions.Timeout as e:
                wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                if attempt < max_retries - 1:
                    logger.warning(f"⏱️ Timeout intento {attempt + 1}/{max_retries} ({lat:.2f}, {lon:.2f}). Reintentando en {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"❌ Timeout después de {max_retries} intentos ({lat:.2f}, {lon:.2f})")
                    return None

            except Exception as e:
                logger.error(f"❌ Error fetching historical weather: {e}")
                return None

        return None

    def fetch_forecast_weather(
        self,
        lat: float,
        lon: float,
        days: int = 16
    ) -> pd.DataFrame:
        """
        Obtiene forecast meteorológico (próximos días)

        Parameters:
        -----------
        lat, lon : float
            Coordenadas
        days : int
            Días de forecast (máx 16)

        Returns:
        --------
        pd.DataFrame : Forecast diario
        """
        variables = [
            'temperature_2m_mean',
            'temperature_2m_max',
            'temperature_2m_min',
            'precipitation_sum',
            'sunshine_duration'
        ]

        # Retry con exponential backoff
        import time
        max_retries = 3

        params = {
            'latitude': lat,
            'longitude': lon,
            'daily': ','.join(variables),
            'forecast_days': min(days, 16),
            'timezone': 'Europe/Madrid'
        }

        # Añadir API key si está disponible
        if self.api_key:
            params['apikey'] = self.api_key

        for attempt in range(max_retries):
            try:
                response = requests.get(self.forecast_url, params=params, timeout=60)
                response.raise_for_status()
                data = response.json()

                df = pd.DataFrame(data['daily'])
                df['time'] = pd.to_datetime(df['time'])
                df = df.rename(columns={'time': 'date'})

                return df

            except requests.exceptions.Timeout as e:
                wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                if attempt < max_retries - 1:
                    logger.warning(f"⏱️ Timeout intento {attempt + 1}/{max_retries} ({lat:.2f}, {lon:.2f}). Reintentando en {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"❌ Timeout después de {max_retries} intentos ({lat:.2f}, {lon:.2f})")
                    return None

            except Exception as e:
                logger.error(f"❌ Error fetching forecast: {e}")
                return None

        return None

    def calculate_temporal_features(
        self,
        weather_df: pd.DataFrame,
        target_date: datetime,
        windows: Optional[Dict[str, List[int]]] = None
    ) -> Dict[str, float]:
        """
        Calcula features temporales agregadas para una fecha objetivo

        Parameters:
        -----------
        weather_df : pd.DataFrame
            Datos meteorológicos (debe incluir target_date y días anteriores)
        target_date : datetime
            Fecha objetivo
        windows : dict, optional
            Ventanas temporales para agregación
            Default: {'precipitation': [7, 15, 20], 'temperature': [7, 15], ...}

        Returns:
        --------
        dict : Features temporales
            Ej: {'precip_sum_7d': 45.2, 'temp_mean_7d': 12.5, ...}
        """
        if windows is None:
            windows = config.TEMPORAL_FEATURES

        # Asegurar que weather_df está ordenado por fecha
        weather_df = weather_df.sort_values('date')

        # Filtrar hasta target_date
        weather_past = weather_df[weather_df['date'] <= target_date].copy()

        if len(weather_past) == 0:
            logger.warning(f"No hay datos meteorológicos para {target_date}")
            return {}

        features = {}

        # ========== PRECIPITACIÓN ==========
        for window in windows.get('precipitation_windows', [7, 15, 20]):
            start_date = target_date - timedelta(days=window)
            window_data = weather_past[weather_past['date'] >= start_date]

            if len(window_data) > 0:
                # Suma acumulada
                precip_sum = window_data['precipitation_sum'].sum()
                features[f'precip_sum_{window}d'] = precip_sum

                # Máximo diario
                precip_max = window_data['precipitation_sum'].max()
                features[f'precip_max_{window}d'] = precip_max

                # Días con lluvia >5mm
                rain_days = (window_data['precipitation_sum'] > 5.0).sum()
                features[f'rain_days_{window}d'] = rain_days
            else:
                features[f'precip_sum_{window}d'] = 0
                features[f'precip_max_{window}d'] = 0
                features[f'rain_days_{window}d'] = 0

        # Días desde última lluvia significativa
        rain_threshold = config.TEMPORAL_FEATURES.get('rain_threshold_mm', 5.0)
        rainy_days = weather_past[weather_past['precipitation_sum'] > rain_threshold]
        if len(rainy_days) > 0:
            last_rain_date = rainy_days['date'].max()
            days_since_rain = (target_date - last_rain_date).days
        else:
            days_since_rain = 999  # Sin lluvia en el registro
        features['days_since_rain'] = days_since_rain

        # ========== TEMPERATURA ==========
        for window in windows.get('temperature_windows', [7, 15]):
            start_date = target_date - timedelta(days=window)
            window_data = weather_past[weather_past['date'] >= start_date]

            if len(window_data) > 0:
                temp_mean = window_data['temperature_2m_mean'].mean()
                temp_min = window_data['temperature_2m_min'].min()
                temp_max = window_data['temperature_2m_max'].max()

                features[f'temp_mean_{window}d'] = temp_mean
                features[f'temp_min_{window}d'] = temp_min
                features[f'temp_max_{window}d'] = temp_max
            else:
                features[f'temp_mean_{window}d'] = np.nan
                features[f'temp_min_{window}d'] = np.nan
                features[f'temp_max_{window}d'] = np.nan

        # ========== SUNSHINE/RADIACIÓN ==========
        for window in windows.get('sunshine_windows', [7, 15, 20]):
            start_date = target_date - timedelta(days=window)
            window_data = weather_past[weather_past['date'] >= start_date]

            if len(window_data) > 0 and 'sunshine_duration' in window_data.columns:
                # Sunshine en horas
                sunshine_sum = window_data['sunshine_duration'].sum() / 3600
                features[f'sunshine_hours_{window}d'] = sunshine_sum
            else:
                features[f'sunshine_hours_{window}d'] = 0

        # ========== HUMEDAD DEL SUELO ==========
        if 'soil_moisture_0_to_7cm' in weather_past.columns:
            recent_soil = weather_past.tail(7)['soil_moisture_0_to_7cm'].mean()
            features['soil_moisture_7d_mean'] = recent_soil
        else:
            features['soil_moisture_7d_mean'] = np.nan

        return features

    def get_weather_for_observations(
        self,
        observations_df: pd.DataFrame,
        date_col: str = 'date',
        lat_col: str = 'lat',
        lon_col: str = 'lon'
    ) -> pd.DataFrame:
        """
        Obtiene datos meteorológicos para cada observación

        Parámetros críticos:
        - Para cada observación, obtiene meteo de esa fecha + 30 días anteriores
        - Calcula features temporales

        Parameters:
        -----------
        observations_df : pd.DataFrame
            Observaciones con fecha y coordenadas

        Returns:
        --------
        pd.DataFrame : Observaciones con features meteorológicas añadidas
        """
        logger.info(f"Obteniendo datos meteorológicos para {len(observations_df)} observaciones...")

        obs_with_meteo = []

        # Agrupar por ubicación para minimizar requests
        grouped = observations_df.groupby([lat_col, lon_col])

        for (lat, lon), group in grouped:
            # Rango de fechas para esta ubicación
            min_date = group[date_col].min() - timedelta(days=30)
            max_date = group[date_col].max()

            # Fetch weather
            weather_df = self.fetch_historical_weather(lat, lon, min_date, max_date)

            if weather_df is None:
                logger.warning(f"No se pudo obtener meteo para ({lat:.4f}, {lon:.4f})")
                continue

            # Para cada observación en esta ubicación
            for idx, obs in group.iterrows():
                obs_date = obs[date_col]

                # Calcular features temporales
                meteo_features = self.calculate_temporal_features(weather_df, obs_date)

                # Combinar con observación
                obs_dict = obs.to_dict()
                obs_dict.update(meteo_features)

                obs_with_meteo.append(obs_dict)

            logger.info(f"  Procesadas {len(group)} obs en ({lat:.4f}, {lon:.4f})")

        result_df = pd.DataFrame(obs_with_meteo)
        logger.info(f"✅ {len(result_df)} observaciones con datos meteorológicos")

        return result_df

    def get_weather_for_grid(
        self,
        grid_df: pd.DataFrame,
        target_date: datetime,
        use_forecast: bool = False,
        sample_resolution_deg: float = 0.5
    ) -> pd.DataFrame:
        """
        Obtiene datos meteorológicos para todas las celdas del grid en una fecha

        OPTIMIZACIÓN: Usa interpolación espacial inteligente
        - Hace API calls solo cada ~50km (sample_resolution_deg)
        - Interpola vectorialmente al resto del grid
        - 900k celdas en ~5 minutos vs ~750 horas

        Parameters:
        -----------
        grid_df : pd.DataFrame
            Grid con columnas ['lat', 'lon', ...]
        target_date : datetime
            Fecha objetivo
        use_forecast : bool
            Si True, usa forecast API (para fechas futuras)
        sample_resolution_deg : float
            Resolución del sampling (grados). Default 0.5° ≈ 50km

        Returns:
        --------
        pd.DataFrame : Grid con features meteorológicas interpoladas
        """
        logger.info(f"\n🌧️ Obteniendo meteo para {len(grid_df):,} celdas")
        logger.info(f"   Fecha: {target_date.date()}")
        logger.info(f"   Resolución sampling: {sample_resolution_deg}° (~{sample_resolution_deg*111:.0f}km)")

        # ========== PASO 1: SAMPLEAR GRID ==========
        lat_min, lat_max = grid_df['lat'].min(), grid_df['lat'].max()
        lon_min, lon_max = grid_df['lon'].min(), grid_df['lon'].max()

        lat_samples = np.arange(lat_min, lat_max + sample_resolution_deg, sample_resolution_deg)
        lon_samples = np.arange(lon_min, lon_max + sample_resolution_deg, sample_resolution_deg)

        n_samples = len(lat_samples) * len(lon_samples)
        logger.info(f"   Puntos de muestreo: {n_samples}")

        # ========== PASO 2: FETCH METEO PARA SAMPLES ==========
        logger.info(f"\n   📥 Descargando datos meteorológicos...")

        meteo_samples = []
        sample_count = 0

        import time

        for lat in lat_samples:
            for lon in lon_samples:
                sample_count += 1

                # Progress
                if sample_count % 10 == 0 or sample_count == n_samples:
                    logger.info(f"      {sample_count}/{n_samples} samples ({100*sample_count/n_samples:.1f}%)")

                # Fetch
                if use_forecast:
                    weather_df = self.fetch_forecast_weather(lat, lon)
                else:
                    start_date = target_date - timedelta(days=30)
                    end_date = target_date
                    weather_df = self.fetch_historical_weather(lat, lon, start_date, end_date)

                if weather_df is not None:
                    meteo_features = self.calculate_temporal_features(weather_df, target_date)
                    meteo_features['lat'] = lat
                    meteo_features['lon'] = lon
                    meteo_samples.append(meteo_features)

                # Delay para evitar rate limiting (API gratis de Open-Meteo)
                # Permite ~300 requests/hora = 5/min = 1 cada 12s para ser conservadores
                # Usamos 0.3s para balance entre velocidad y límites
                if sample_count < n_samples:  # No esperar después del último
                    time.sleep(0.3)

        if len(meteo_samples) == 0:
            logger.error("❌ No se pudo obtener datos meteorológicos")
            return None

        meteo_df = pd.DataFrame(meteo_samples)
        logger.info(f"   ✅ {len(meteo_df)} puntos de muestra obtenidos")

        # ========== PASO 3: INTERPOLACIÓN VECTORIZADA ==========
        logger.info(f"\n   🔄 Interpolando a {len(grid_df):,} celdas del grid...")

        from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

        # Coordenadas de los samples
        sample_coords = meteo_df[['lat', 'lon']].values

        # Coordenadas del grid completo
        grid_coords = grid_df[['lat', 'lon']].values

        # Interpolar cada feature
        grid_with_meteo = grid_df.copy()

        meteo_features = [col for col in meteo_df.columns if col not in ['lat', 'lon']]

        logger.info(f"   Features a interpolar: {len(meteo_features)}")

        for i, feature in enumerate(meteo_features):
            if (i + 1) % 5 == 0 or (i + 1) == len(meteo_features):
                logger.info(f"      {i+1}/{len(meteo_features)} features ({100*(i+1)/len(meteo_features):.1f}%)")

            sample_values = meteo_df[feature].values

            # Usar interpolación lineal con nearest neighbor como fallback
            try:
                # Linear interpolator
                linear_interp = LinearNDInterpolator(sample_coords, sample_values)
                interpolated = linear_interp(grid_coords)

                # Nearest neighbor para puntos fuera del convex hull
                nans = np.isnan(interpolated)
                if nans.any():
                    nearest_interp = NearestNDInterpolator(sample_coords, sample_values)
                    interpolated[nans] = nearest_interp(grid_coords[nans])

                grid_with_meteo[feature] = interpolated

            except Exception as e:
                logger.warning(f"      Error interpolando {feature}: {e}. Usando nearest neighbor.")
                # Fallback: nearest neighbor
                nearest_interp = NearestNDInterpolator(sample_coords, sample_values)
                grid_with_meteo[feature] = nearest_interp(grid_coords)

        logger.info(f"\n   ✅ Interpolación completada")
        logger.info(f"   📊 {len(meteo_features)} features meteorológicas añadidas")

        # Validación: check for NaNs
        nan_count = grid_with_meteo[meteo_features].isnull().sum().sum()
        if nan_count > 0:
            logger.warning(f"   ⚠️ {nan_count} valores NaN detectados (serán imputados)")

        return grid_with_meteo
