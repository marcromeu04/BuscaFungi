"""
BuscaFungi - Grid Management
Sistema de grid fijo y determinístico para evitar data leakage
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Tuple

from . import config

logger = logging.getLogger(__name__)


class GridManager:
    """
    Gestor de grid fijo y determinístico

    El grid se genera una vez y se mantiene constante entre entrenamientos
    y predicciones, evitando data leakage.
    """

    def __init__(self, resolution_km=None, bounds=None):
        """
        Parameters:
        -----------
        resolution_km : float, optional
            Resolución del grid en km (default: config.GRID_RESOLUTION_KM)
        bounds : dict, optional
            Límites geográficos {'lat_min', 'lat_max', 'lon_min', 'lon_max'}
        """
        self.resolution_km = resolution_km or config.GRID_RESOLUTION_KM
        self.bounds = bounds or config.SPAIN_BOUNDS

        self.grid_df = None
        self.cell_lookup = {}  # Para búsqueda rápida de celdas

        logger.info(f"GridManager inicializado: {self.resolution_km}km, región: {config.FOCUS_REGION}")

    def create_grid(self, use_sample=None, sample_size=None):
        """
        Crea el grid determinístico

        Parameters:
        -----------
        use_sample : bool, optional
            Si True, devuelve una muestra aleatoria (pero con seed fijo)
        sample_size : int, optional
            Tamaño de la muestra

        Returns:
        --------
        pd.DataFrame : Grid con columnas ['cell_id', 'lat', 'lon']
        """
        logger.info("Creando grid deterministico...")

        # Calcular step size
        # 1 grado latitud ≈ 111.32 km
        # 1 grado longitud varía con latitud: 111.32 * cos(lat) km
        lat_step = self.resolution_km / 111.32

        # Usar latitud media para calcular lon_step
        lat_mean = (self.bounds['lat_min'] + self.bounds['lat_max']) / 2
        lon_step = self.resolution_km / (111.32 * np.cos(np.radians(lat_mean)))

        # Generar coordenadas
        lats = np.arange(
            self.bounds['lat_min'],
            self.bounds['lat_max'],
            lat_step
        )
        lons = np.arange(
            self.bounds['lon_min'],
            self.bounds['lon_max'],
            lon_step
        )

        # Crear meshgrid
        lon_grid, lat_grid = np.meshgrid(lons, lats)

        # Flatten
        lats_flat = lat_grid.flatten()
        lons_flat = lon_grid.flatten()

        # Crear DataFrame
        self.grid_df = pd.DataFrame({
            'lat': lats_flat,
            'lon': lons_flat
        })

        # Crear cell_id único y determinístico basado en lat/lon redondeados
        self.grid_df['cell_id'] = self.grid_df.apply(
            lambda row: f"{row['lat']:.6f}_{row['lon']:.6f}",
            axis=1
        )

        # Reordenar columnas
        self.grid_df = self.grid_df[['cell_id', 'lat', 'lon']]

        total_cells = len(self.grid_df)
        logger.info(f"Grid creado: {total_cells:,} celdas")

        # Sampling si se requiere (con seed fijo para reproducibilidad)
        use_sample = use_sample if use_sample is not None else config.USE_SAMPLE
        sample_size = sample_size or config.SAMPLE_SIZE

        if use_sample and total_cells > sample_size:
            logger.info(f"Muestreando {sample_size:,} celdas (seed fijo)...")
            self.grid_df = self.grid_df.sample(
                n=sample_size,
                random_state=42  # SEED FIJO = MUESTRA DETERMINÍSTICA
            ).reset_index(drop=True)
            logger.info(f"Muestra final: {len(self.grid_df):,} celdas")

        # Crear lookup table para búsqueda rápida
        self._build_lookup_table()

        return self.grid_df

    def _build_lookup_table(self):
        """
        Construye tabla de búsqueda para mapeo rápido de observaciones a celdas
        """
        logger.info("Construyendo lookup table...")

        for idx, row in self.grid_df.iterrows():
            # Usar lat/lon redondeados como key
            key = (round(row['lat'], 6), round(row['lon'], 6))
            self.cell_lookup[key] = {
                'index': idx,
                'cell_id': row['cell_id'],
                'lat': row['lat'],
                'lon': row['lon']
            }

        logger.info(f"Lookup table: {len(self.cell_lookup):,} entries")

    def snap_to_grid(self, lat, lon, tolerance_km=5.0):
        """
        Encuentra la celda del grid más cercana a un punto (lat, lon)

        CRÍTICO PARA EVITAR DATA LEAKAGE:
        Las observaciones deben asignarse a celdas del grid de forma determinística

        Parameters:
        -----------
        lat, lon : float
            Coordenadas del punto
        tolerance_km : float
            Distancia máxima permitida (km). Si no hay celda cercana, return None

        Returns:
        --------
        dict or None : {'index': int, 'cell_id': str, 'lat': float, 'lon': float}
        """
        # Primero intentar lookup directo (rápido)
        key = (round(lat, 6), round(lon, 6))
        if key in self.cell_lookup:
            return self.cell_lookup[key]

        # Si no está exacto, buscar la celda más cercana
        distances = np.sqrt(
            (self.grid_df['lat'] - lat)**2 +
            (self.grid_df['lon'] - lon)**2
        )

        min_idx = distances.idxmin()
        min_dist_deg = distances.iloc[min_idx]

        # Convertir a km (aproximado)
        min_dist_km = min_dist_deg * 111.32

        if min_dist_km > tolerance_km:
            logger.warning(
                f"Punto ({lat:.4f}, {lon:.4f}) está a {min_dist_km:.1f}km "
                f"de la celda más cercana (tolerancia: {tolerance_km}km)"
            )
            return None

        row = self.grid_df.iloc[min_idx]
        return {
            'index': min_idx,
            'cell_id': row['cell_id'],
            'lat': row['lat'],
            'lon': row['lon']
        }

    def assign_observations_to_grid(self, observations_df):
        """
        Asigna observaciones a celdas del grid

        IMPORTANTE: Esto es determinístico y reproducible

        Parameters:
        -----------
        observations_df : pd.DataFrame
            DataFrame con columnas ['lat', 'lon', ...] (observaciones)

        Returns:
        --------
        pd.DataFrame : Observaciones con columnas adicionales:
            - cell_id: ID de la celda asignada
            - grid_lat, grid_lon: Coordenadas de la celda
            - snap_distance_km: Distancia a la celda
        """
        logger.info(f"Asignando {len(observations_df)} observaciones al grid...")

        obs_with_grid = []
        skipped = 0

        for idx, obs in observations_df.iterrows():
            cell = self.snap_to_grid(obs['lat'], obs['lon'])

            if cell is None:
                skipped += 1
                continue

            obs_dict = obs.to_dict()
            obs_dict['cell_id'] = cell['cell_id']
            obs_dict['grid_lat'] = cell['lat']
            obs_dict['grid_lon'] = cell['lon']

            # Calcular distancia de snap
            snap_dist = np.sqrt(
                (obs['lat'] - cell['lat'])**2 +
                (obs['lon'] - cell['lon'])**2
            ) * 111.32
            obs_dict['snap_distance_km'] = snap_dist

            obs_with_grid.append(obs_dict)

        result_df = pd.DataFrame(obs_with_grid)

        logger.info(
            f"Asignadas: {len(result_df)}/{len(observations_df)} "
            f"({skipped} fuera de tolerancia)"
        )

        if len(result_df) > 0:
            logger.info(
                f"Distancia snap promedio: {result_df['snap_distance_km'].mean():.2f}km "
                f"(max: {result_df['snap_distance_km'].max():.2f}km)"
            )

        return result_df

    def get_grid_bounds(self):
        """
        Retorna límites del grid actual
        """
        if self.grid_df is None:
            return None

        return {
            'lat_min': self.grid_df['lat'].min(),
            'lat_max': self.grid_df['lat'].max(),
            'lon_min': self.grid_df['lon'].min(),
            'lon_max': self.grid_df['lon'].max(),
            'n_cells': len(self.grid_df)
        }

    def save_grid(self, filepath):
        """
        Guarda el grid a CSV para reutilización
        """
        if self.grid_df is None:
            logger.error("No hay grid para guardar")
            return

        self.grid_df.to_csv(filepath, index=False)
        logger.info(f"Grid guardado: {filepath}")

    def load_grid(self, filepath):
        """
        Carga grid desde CSV
        """
        self.grid_df = pd.read_csv(filepath)
        self._build_lookup_table()
        logger.info(f"Grid cargado: {filepath} ({len(self.grid_df):,} celdas)")
        return self.grid_df
