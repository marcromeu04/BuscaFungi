"""
BuscaFungi - Pseudo-Absence Generation
Generación inteligente de pseudo-ausencias espaciales y ecológicas
"""

import numpy as np
import pandas as pd
import logging
from typing import Optional

from . import config
from .utils import haversine_distance

logger = logging.getLogger(__name__)


def generate_smart_pseudo_absences(
    presences_df: pd.DataFrame,
    grid_df: pd.DataFrame,
    ratio: float = 2.0,
    min_distance_km: float = 10.0,
    cluster_col: Optional[str] = 'cluster',
    use_cluster_filter: bool = True
) -> list:
    """
    Genera pseudo-ausencias inteligentes usando criterios espaciales Y ecológicos

    ESTRATEGIA:
    1. Ausencias deben estar LEJOS de presencias (espacial)
    2. Ausencias deben estar en CLUSTERS DIFERENTES (ecológico)
    3. Ausencias deben ser representativas del "background" disponible

    Parameters:
    -----------
    presences_df : pd.DataFrame
        DataFrame con presencias (debe tener 'lat', 'lon')
    grid_df : pd.DataFrame
        Grid completo con features (debe tener 'lat', 'lon')
    ratio : float
        Ratio ausencias:presencias (default 2.0 = 2 ausencias por presencia)
    min_distance_km : float
        Distancia mínima a cualquier presencia (km)
    cluster_col : str, optional
        Columna de cluster en grid_df (para filtro ecológico)
    use_cluster_filter : bool
        Si True, prefiere ausencias en clusters sin presencias

    Returns:
    --------
    list : Índices del grid_df seleccionados como pseudo-ausencias
    """
    n_presences = len(presences_df)
    n_absences_needed = int(n_presences * ratio)

    logger.info(f"Generando pseudo-ausencias:")
    logger.info(f"  Presencias: {n_presences}")
    logger.info(f"  Ausencias necesarias: {n_absences_needed} (ratio: {ratio})")
    logger.info(f"  Distancia mínima: {min_distance_km}km")

    # Usar grid_lat, grid_lon si existen (observaciones snapped), sino lat/lon
    if 'grid_lat' in presences_df.columns:
        presence_lats = presences_df['grid_lat'].values
        presence_lons = presences_df['grid_lon'].values
    else:
        presence_lats = presences_df['lat'].values
        presence_lons = presences_df['lon'].values

    # 1. FILTRO ESPACIAL: Eliminar celdas cercanas a presencias
    logger.info("  Aplicando filtro espacial...")

    grid_indices = grid_df.index.tolist()
    valid_absence_indices = []

    for idx in grid_indices:
        cell_lat = grid_df.at[idx, 'lat']
        cell_lon = grid_df.at[idx, 'lon']

        # Calcular distancia a TODAS las presencias
        min_dist_to_presence = np.inf

        for p_lat, p_lon in zip(presence_lats, presence_lons):
            dist = haversine_distance(cell_lat, cell_lon, p_lat, p_lon)
            if dist < min_dist_to_presence:
                min_dist_to_presence = dist

        # Mantener solo si está suficientemente lejos
        if min_dist_to_presence >= min_distance_km:
            valid_absence_indices.append(idx)

    logger.info(f"    {len(valid_absence_indices)} celdas lejanas de presencias")

    if len(valid_absence_indices) < n_absences_needed:
        logger.warning(
            f"Solo {len(valid_absence_indices)} celdas válidas "
            f"(necesarias: {n_absences_needed}). "
            f"Reduciendo distancia mínima..."
        )
        # Retry con distancia reducida
        return generate_smart_pseudo_absences(
            presences_df,
            grid_df,
            ratio=ratio,
            min_distance_km=min_distance_km * 0.7,
            cluster_col=cluster_col,
            use_cluster_filter=use_cluster_filter
        )

    # 2. FILTRO ECOLÓGICO (OPCIONAL): Preferir clusters diferentes
    if use_cluster_filter and cluster_col in grid_df.columns and cluster_col in presences_df.columns:
        logger.info("  Aplicando filtro ecológico (clusters)...")

        # Clusters donde hay presencias
        presence_clusters = set(presences_df[cluster_col].unique())
        logger.info(f"    Clusters con presencias: {presence_clusters}")

        # Preferir celdas en clusters SIN presencias
        absence_in_other_clusters = [
            idx for idx in valid_absence_indices
            if grid_df.at[idx, cluster_col] not in presence_clusters
        ]

        if len(absence_in_other_clusters) >= n_absences_needed:
            logger.info(f"    {len(absence_in_other_clusters)} celdas en clusters diferentes")
            valid_absence_indices = absence_in_other_clusters
        else:
            logger.info(
                f"    Solo {len(absence_in_other_clusters)} en otros clusters. "
                f"Usando todas las celdas válidas."
            )

    # 3. MUESTREO ALEATORIO (con seed para reproducibilidad)
    if len(valid_absence_indices) > n_absences_needed:
        np.random.seed(42)  # Seed fijo = reproducibilidad
        selected_indices = np.random.choice(
            valid_absence_indices,
            size=n_absences_needed,
            replace=False
        )
    else:
        selected_indices = valid_absence_indices

    logger.info(f"  ✅ {len(selected_indices)} pseudo-ausencias generadas")

    # 4. VALIDACIÓN: Verificar que no hay presencias en las ausencias seleccionadas
    # (Por si acaso hay overlaps debido a snap to grid)
    if 'cell_id' in presences_df.columns and 'cell_id' in grid_df.columns:
        presence_cell_ids = set(presences_df['cell_id'].unique())
        selected_cell_ids = set(grid_df.loc[selected_indices, 'cell_id'])

        overlap = presence_cell_ids.intersection(selected_cell_ids)
        if len(overlap) > 0:
            logger.warning(f"  ⚠️ {len(overlap)} ausencias overlap con presencias. Removiendo...")
            selected_indices = [
                idx for idx in selected_indices
                if grid_df.at[idx, 'cell_id'] not in presence_cell_ids
            ]
            logger.info(f"  ✅ {len(selected_indices)} ausencias limpias finales")

    return list(selected_indices)


def validate_absence_quality(
    absence_indices: list,
    presences_df: pd.DataFrame,
    grid_df: pd.DataFrame
) -> dict:
    """
    Valida la calidad de las pseudo-ausencias generadas

    Retorna estadísticas sobre:
    - Distancia promedio a presencias
    - Distribución espacial
    - Diversidad de clusters

    Parameters:
    -----------
    absence_indices : list
        Índices de ausencias en grid_df
    presences_df : pd.DataFrame
        DataFrame de presencias
    grid_df : pd.DataFrame
        Grid completo

    Returns:
    --------
    dict : Estadísticas de calidad
    """
    absences_df = grid_df.loc[absence_indices]

    # Calcular distancia mínima de cada ausencia a presencias
    min_distances = []

    for idx, absence in absences_df.iterrows():
        distances = []
        for _, presence in presences_df.iterrows():
            dist = haversine_distance(
                absence['lat'], absence['lon'],
                presence.get('grid_lat', presence['lat']),
                presence.get('grid_lon', presence['lon'])
            )
            distances.append(dist)

        min_distances.append(min(distances))

    stats = {
        'n_absences': len(absence_indices),
        'n_presences': len(presences_df),
        'ratio': len(absence_indices) / len(presences_df),
        'min_distance_km_mean': np.mean(min_distances),
        'min_distance_km_std': np.std(min_distances),
        'min_distance_km_min': np.min(min_distances),
        'min_distance_km_max': np.max(min_distances),
    }

    # Distribución de clusters (si existe)
    if 'cluster' in absences_df.columns:
        absence_clusters = absences_df['cluster'].value_counts().to_dict()
        stats['cluster_distribution'] = absence_clusters

        if 'cluster' in presences_df.columns:
            presence_clusters = set(presences_df['cluster'].unique())
            absence_clusters_set = set(absences_df['cluster'].unique())
            overlap = presence_clusters.intersection(absence_clusters_set)

            stats['clusters_with_presences'] = len(presence_clusters)
            stats['clusters_with_absences'] = len(absence_clusters_set)
            stats['clusters_overlap'] = len(overlap)

    logger.info("Calidad de pseudo-ausencias:")
    logger.info(f"  Distancia mínima promedio: {stats['min_distance_km_mean']:.1f}km")
    logger.info(f"  Rango: {stats['min_distance_km_min']:.1f} - {stats['min_distance_km_max']:.1f}km")

    return stats


def temporal_pseudo_absences(
    presences_df: pd.DataFrame,
    grid_df: pd.DataFrame,
    temporal_col: str = 'date'
) -> list:
    """
    Genera pseudo-ausencias TEMPORALES:
    - Mismas ubicaciones que presencias, pero en fechas diferentes (sin observaciones)

    Esto captura: "Esta zona es adecuada, pero NO se encontraron hongos en estas fechas"

    Útil para modelar variabilidad temporal

    Parameters:
    -----------
    presences_df : pd.DataFrame
        Presencias con fecha
    grid_df : pd.DataFrame
        Grid con features

    Returns:
    --------
    list : Índices de grid_df o nuevas filas con ausencias temporales
    """
    logger.info("Generando pseudo-ausencias temporales...")

    # TODO: Implementar si se quiere modelar variabilidad temporal explícita
    # Por ahora, el modelo captura temporalidad via features meteorológicas

    logger.warning("Pseudo-ausencias temporales no implementadas aún")
    return []
