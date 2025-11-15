#!/usr/bin/env python3
"""
BuscaFungi - Setup Grid + Clustering
Crea grid con features interpoladas y clustering ecológico

Ejecutar 1 vez para generar grid_clustered.parquet
Tiempo estimado: ~30 minutos
"""

import sys
sys.path.insert(0, 'src')

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from src import config
from src.grid import GridManager
from src.features import FeatureExtractor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_sample_grid(full_grid_df, sample_resolution_km):
    """
    Crea grid de muestreo espaciado para extracción de features

    Parameters:
    -----------
    full_grid_df : pd.DataFrame
        Grid completo
    sample_resolution_km : float
        Resolución del sampling (ej: 10km)

    Returns:
    --------
    pd.DataFrame : Grid de samples
    """
    logger.info(f"\n📍 Creando grid de muestreo ({sample_resolution_km}km)...")

    lat_min, lat_max = full_grid_df['lat'].min(), full_grid_df['lat'].max()
    lon_min, lon_max = full_grid_df['lon'].min(), full_grid_df['lon'].max()

    # Step en grados
    lat_step = sample_resolution_km / 111.32
    lon_mean = (lat_min + lat_max) / 2
    lon_step = sample_resolution_km / (111.32 * np.cos(np.radians(lon_mean)))

    # Crear grid de samples
    lats = np.arange(lat_min, lat_max + lat_step, lat_step)
    lons = np.arange(lon_min, lon_max + lon_step, lon_step)

    sample_data = []
    for lat in lats:
        for lon in lons:
            sample_data.append({
                'lat': lat,
                'lon': lon,
                'cell_id': f"sample_{lat:.6f}_{lon:.6f}"
            })

    sample_grid = pd.DataFrame(sample_data)

    logger.info(f"   ✅ Grid de muestreo: {len(sample_grid):,} puntos")
    logger.info(f"   📏 Cobertura: {len(lats)} lats × {len(lons)} lons")

    return sample_grid


def extract_features_for_samples(sample_grid_df):
    """
    Extrae features ambientales para puntos de muestreo

    Usa APIs reales pero solo para ~250 puntos
    """
    logger.info(f"\n🌱 Extrayendo features para {len(sample_grid_df):,} samples...")
    logger.info(f"   ⏱️ Tiempo estimado: ~{len(sample_grid_df) * 3} segundos")

    feature_extractor = FeatureExtractor()

    features_list = []

    for idx, row in sample_grid_df.iterrows():
        if idx % 10 == 0:
            pct = 100 * idx / len(sample_grid_df)
            logger.info(f"   📍 {idx}/{len(sample_grid_df)} ({pct:.1f}%) - lat={row['lat']:.2f}, lon={row['lon']:.2f}")

        # Extraer features ambientales (suelo, elevación, vegetación)
        env_features = feature_extractor.extract_environmental_features(
            row['lat'],
            row['lon']
        )

        features = {
            'cell_id': row['cell_id'],
            'lat': row['lat'],
            'lon': row['lon']
        }
        features.update(env_features)

        features_list.append(features)

    features_df = pd.DataFrame(features_list)

    # One-hot encoding para vegetación
    if 'vegetation_type' in features_df.columns:
        veg_dummies = pd.get_dummies(features_df['vegetation_type'], prefix='veg')
        features_df = pd.concat([features_df, veg_dummies], axis=1)

    logger.info(f"\n   ✅ Features extraídas: {len(features_df.columns)} columnas")

    return features_df


def interpolate_features_to_grid(sample_features_df, full_grid_df):
    """
    Interpola features desde samples al grid completo

    Usa interpolación lineal + nearest neighbor como fallback
    """
    logger.info(f"\n🔄 Interpolando features a {len(full_grid_df):,} celdas...")

    # Coordenadas
    sample_coords = sample_features_df[['lat', 'lon']].values
    grid_coords = full_grid_df[['lat', 'lon']].values

    # Inicializar grid con coordenadas
    interpolated_grid = full_grid_df.copy()

    # Features a interpolar (excluir metadata)
    feature_cols = [
        col for col in sample_features_df.columns
        if col not in ['cell_id', 'lat', 'lon', 'vegetation_type']
    ]

    logger.info(f"   Features a interpolar: {len(feature_cols)}")

    for i, feature in enumerate(feature_cols):
        if (i + 1) % 5 == 0 or (i + 1) == len(feature_cols):
            logger.info(f"      {i+1}/{len(feature_cols)} features ({100*(i+1)/len(feature_cols):.1f}%)")

        sample_values = sample_features_df[feature].values

        try:
            # Linear interpolation
            linear_interp = LinearNDInterpolator(sample_coords, sample_values)
            interpolated = linear_interp(grid_coords)

            # Nearest neighbor para NaNs (fuera del convex hull)
            nans = np.isnan(interpolated)
            if nans.any():
                nearest_interp = NearestNDInterpolator(sample_coords, sample_values)
                interpolated[nans] = nearest_interp(grid_coords[nans])

            interpolated_grid[feature] = interpolated

        except Exception as e:
            logger.warning(f"      Error interpolando {feature}: {e}. Usando nearest neighbor.")
            nearest_interp = NearestNDInterpolator(sample_coords, sample_values)
            interpolated_grid[feature] = nearest_interp(grid_coords)

    logger.info(f"\n   ✅ Interpolación completada")

    # Validar NaNs
    nan_count = interpolated_grid[feature_cols].isnull().sum().sum()
    if nan_count > 0:
        logger.warning(f"   ⚠️ {nan_count} NaN detectados. Imputando con mediana...")
        interpolated_grid[feature_cols] = interpolated_grid[feature_cols].fillna(
            interpolated_grid[feature_cols].median()
        )

    return interpolated_grid


def perform_clustering(grid_with_features_df, n_components=15):
    """
    Clustering GMM sobre features ambientales

    Identifica nichos ecológicos similares
    """
    logger.info(f"\n🎲 Clustering GMM (n_components={n_components})...")

    # Features para clustering (solo ambientales, no temporales)
    cluster_features = [
        'ph', 'elevation', 'twi', 'organic_carbon', 'slope',
        'clay_percent', 'sand_percent', 'aspect_sin', 'aspect_cos'
    ]

    # Añadir features de vegetación
    veg_cols = [col for col in grid_with_features_df.columns if col.startswith('veg_')]
    cluster_features.extend(veg_cols)

    # Filtrar solo columnas que existen
    cluster_features = [
        col for col in cluster_features
        if col in grid_with_features_df.columns
    ]

    logger.info(f"   Features para clustering: {len(cluster_features)}")

    X = grid_with_features_df[cluster_features].copy()

    # Imputar NaNs
    if X.isnull().any().any():
        logger.warning("   Imputando NaNs con mediana...")
        X = X.fillna(X.median())

    # Escalar
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit GMM
    logger.info("   Entrenando GMM...")
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type='full',
        random_state=42,
        max_iter=200,
        n_init=3,
        verbose=1
    )

    gmm.fit(X_scaled)

    # Predecir clusters
    clusters = gmm.predict(X_scaled)
    cluster_probs = gmm.predict_proba(X_scaled)

    # Añadir al grid
    grid_clustered = grid_with_features_df.copy()
    grid_clustered['cluster'] = clusters
    grid_clustered['cluster_confidence'] = cluster_probs.max(axis=1)

    # Estadísticas
    logger.info(f"\n   📊 Distribución de clusters:")
    cluster_counts = pd.Series(clusters).value_counts().sort_index()

    for cluster_id, count in cluster_counts.items():
        pct = count / len(clusters) * 100
        logger.info(f"      Cluster {cluster_id:2d}: {count:7,} celdas ({pct:5.1f}%)")

    logger.info(f"\n   ✅ Clustering completado")

    return grid_clustered, gmm, scaler, cluster_features


def main():
    """
    Pipeline principal de setup
    """
    print("\n" + "="*70)
    print("🍄 BuscaFungi - Setup Grid + Clustering")
    print("="*70)
    print(f"\nRegión: {config.FOCUS_REGION}")
    print(f"Resolución grid: {config.GRID_RESOLUTION_KM}km")
    print(f"Resolución sampling: {config.FEATURE_SAMPLE_RESOLUTION_KM}km")
    print(f"Tiempo estimado: ~30 minutos")
    print("\n" + "="*70)

    # ========== PASO 1: CREAR GRID ==========
    logger.info("\n" + "="*70)
    logger.info("PASO 1: Crear Grid Base")
    logger.info("="*70)

    grid_manager = GridManager()
    full_grid = grid_manager.create_grid(use_sample=False)

    logger.info(f"✅ Grid creado: {len(full_grid):,} celdas")
    logger.info(f"   Área: {config.SPAIN_BOUNDS}")

    # ========== PASO 2: CREAR SAMPLE GRID ==========
    logger.info("\n" + "="*70)
    logger.info("PASO 2: Crear Sample Grid")
    logger.info("="*70)

    sample_grid = create_sample_grid(
        full_grid,
        config.FEATURE_SAMPLE_RESOLUTION_KM
    )

    # ========== PASO 3: EXTRAER FEATURES PARA SAMPLES ==========
    logger.info("\n" + "="*70)
    logger.info("PASO 3: Extraer Features para Samples")
    logger.info("="*70)

    sample_features = extract_features_for_samples(sample_grid)

    # Guardar samples (para debug)
    output_dir = Path('outputs')
    output_dir.mkdir(exist_ok=True)

    sample_features.to_parquet(output_dir / 'sample_features.parquet', index=False)
    logger.info(f"\n💾 Samples guardados: outputs/sample_features.parquet")

    # ========== PASO 4: INTERPOLAR A GRID COMPLETO ==========
    logger.info("\n" + "="*70)
    logger.info("PASO 4: Interpolar Features al Grid Completo")
    logger.info("="*70)

    grid_with_features = interpolate_features_to_grid(sample_features, full_grid)

    # ========== PASO 5: CLUSTERING ==========
    logger.info("\n" + "="*70)
    logger.info("PASO 5: Clustering Ecológico")
    logger.info("="*70)

    grid_clustered, gmm, scaler, cluster_features = perform_clustering(
        grid_with_features,
        n_components=15  # Optimizar después
    )

    # ========== PASO 6: GUARDAR ==========
    logger.info("\n" + "="*70)
    logger.info("PASO 6: Guardar Resultados")
    logger.info("="*70)

    # Grid clustered
    output_file = output_dir / 'grid_clustered.parquet'
    grid_clustered.to_parquet(output_file, index=False)
    logger.info(f"✅ Grid guardado: {output_file}")
    logger.info(f"   Tamaño: {output_file.stat().st_size / 1024 / 1024:.1f} MB")
    logger.info(f"   Celdas: {len(grid_clustered):,}")
    logger.info(f"   Columnas: {len(grid_clustered.columns)}")

    # Modelo GMM
    import joblib
    gmm_file = output_dir / 'gmm_model.joblib'
    joblib.dump({
        'gmm': gmm,
        'scaler': scaler,
        'cluster_features': cluster_features
    }, gmm_file)
    logger.info(f"✅ GMM guardado: {gmm_file}")

    # Metadata
    metadata = {
        'region': config.FOCUS_REGION,
        'grid_resolution_km': config.GRID_RESOLUTION_KM,
        'sample_resolution_km': config.FEATURE_SAMPLE_RESOLUTION_KM,
        'n_cells': len(grid_clustered),
        'n_samples': len(sample_features),
        'n_clusters': 15,
        'features': list(grid_clustered.columns)
    }

    import json
    with open(output_dir / 'grid_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"✅ Metadata guardado: outputs/grid_metadata.json")

    # ========== RESUMEN ==========
    print("\n" + "="*70)
    print("✅ SETUP COMPLETADO")
    print("="*70)
    print(f"\n📊 Resumen:")
    print(f"   Grid completo: {len(grid_clustered):,} celdas")
    print(f"   Samples usados: {len(sample_features):,} puntos")
    print(f"   Features: {len(cluster_features)} ambientales")
    print(f"   Clusters: 15 nichos ecológicos")
    print(f"\n📁 Archivos generados:")
    print(f"   - outputs/grid_clustered.parquet  (PRINCIPAL)")
    print(f"   - outputs/gmm_model.joblib")
    print(f"   - outputs/sample_features.parquet")
    print(f"   - outputs/grid_metadata.json")
    print(f"\n🎯 Siguiente paso:")
    print(f"   python train.py  (usará grid_clustered.parquet)")
    print("="*70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrumpido por usuario")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
