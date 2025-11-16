#!/usr/bin/env python3
"""
BuscaFungi - Prediction v2
Predice probabilidad de hongos usando cluster features + meteo interpolada

Uso:
    python predict_v2.py --species "Boletus edulis" --date 2024-09-15
    python predict_v2.py --species "Lactarius deliciosus" --date 2024-10-01 --use-forecast
"""

import sys
sys.path.insert(0, 'src')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import logging
import argparse
import joblib

from src import config
from src.meteo import MeteoDataFetcher

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_grid_and_models(species):
    """
    Carga grid pre-procesado y modelo entrenado

    Returns:
    --------
    tuple: (grid_df, model, cluster_features, feature_cols)
    """
    logger.info("\n" + "="*70)
    logger.info("📥 PASO 1: Cargando Grid y Modelo")
    logger.info("="*70)

    # Grid clustered
    grid_file = Path('outputs/grid_clustered.parquet')
    if not grid_file.exists():
        raise FileNotFoundError(
            f"Grid no encontrado: {grid_file}\n"
            f"Ejecuta primero: python setup_grid_clustering.py"
        )

    grid_df = pd.read_parquet(grid_file)
    logger.info(f"✅ Grid cargado: {len(grid_df):,} celdas")
    logger.info(f"   Clusters: {grid_df['cluster'].nunique()}")

    # Modelo
    model_name = species.replace(' ', '_')
    model_file = Path(f'outputs/models/{model_name}_v2.joblib')

    if not model_file.exists():
        raise FileNotFoundError(
            f"Modelo no encontrado: {model_file}\n"
            f"Ejecuta primero: python train_v2.py"
        )

    model_data = joblib.load(model_file)
    model = model_data['model']
    feature_cols = model_data['feature_cols']

    logger.info(f"✅ Modelo cargado: {model_file.name}")
    logger.info(f"   Features: {len(feature_cols)}")

    # Cluster features
    cluster_features_file = Path('outputs/cluster_features.joblib')
    if not cluster_features_file.exists():
        raise FileNotFoundError(
            f"Cluster features no encontradas: {cluster_features_file}\n"
            f"Ejecuta primero: python train_v2.py"
        )

    cluster_features = joblib.load(cluster_features_file)
    logger.info(f"✅ Cluster features cargadas: {len(cluster_features)} clusters")

    return grid_df, model, cluster_features, feature_cols


def get_meteorological_data_for_grid(grid_df, target_date, use_forecast=False):
    """
    Obtiene datos meteorológicos para toda la grid (con interpolación)

    Parameters:
    -----------
    grid_df : pd.DataFrame
        Grid con columnas [cell_id, lat, lon, cluster, ...]
    target_date : datetime
        Fecha objetivo para predicción
    use_forecast : bool
        Si True, usa API forecast para fechas futuras

    Returns:
    --------
    pd.DataFrame : Grid con features meteorológicas añadidas
    """
    logger.info("\n" + "="*70)
    logger.info("📥 PASO 2: Obteniendo Datos Meteorológicos")
    logger.info("="*70)

    logger.info(f"Fecha objetivo: {target_date.date()}")
    logger.info(f"Modo: {'Forecast' if use_forecast else 'Historical'}")

    meteo_fetcher = MeteoDataFetcher(enable_disk_cache=True)

    # Obtener meteo con interpolación
    grid_with_meteo = meteo_fetcher.get_weather_for_grid(
        grid_df,
        target_date=target_date,
        use_forecast=use_forecast,
        sample_resolution_deg=0.5  # ~50km sampling
    )

    if grid_with_meteo is None:
        raise ValueError("No se pudo obtener datos meteorológicos")

    # Validar columnas meteo
    meteo_cols = [col for col in grid_with_meteo.columns
                  if col.startswith(('precip_', 'temp_', 'sunshine_'))]

    logger.info(f"✅ Datos meteorológicos obtenidos")
    logger.info(f"   Features meteo: {len(meteo_cols)}")
    logger.info(f"   Ejemplos: {', '.join(meteo_cols[:3])}...")

    return grid_with_meteo


def prepare_prediction_features(grid_with_meteo, cluster_features, feature_cols):
    """
    Prepara features para predicción: cluster features + meteo

    Parameters:
    -----------
    grid_with_meteo : pd.DataFrame
        Grid con datos meteorológicos
    cluster_features : dict
        {cluster_id: {feature: value}}
    feature_cols : list
        Lista de features esperadas por el modelo

    Returns:
    --------
    pd.DataFrame : Features listas para predicción
    """
    logger.info("\n" + "="*70)
    logger.info("🔧 PASO 3: Preparando Features")
    logger.info("="*70)

    prediction_data = []

    for idx, row in grid_with_meteo.iterrows():
        if idx % 10000 == 0 and idx > 0:
            pct = 100 * idx / len(grid_with_meteo)
            logger.info(f"   📍 {idx:,}/{len(grid_with_meteo):,} celdas ({pct:.1f}%)")

        cluster_id = row['cluster']

        # Features del cluster (ambientales)
        cell_features = cluster_features.get(cluster_id, {}).copy()

        # Añadir features meteorológicas
        for col in grid_with_meteo.columns:
            if col.startswith(('precip_', 'temp_', 'sunshine_')):
                cell_features[col] = row[col]

        # Añadir metadata
        cell_features['cell_id'] = row['cell_id']
        cell_features['lat'] = row['lat']
        cell_features['lon'] = row['lon']
        cell_features['cluster'] = cluster_id

        prediction_data.append(cell_features)

    features_df = pd.DataFrame(prediction_data)

    # Validar que tenemos todas las features necesarias
    missing_cols = set(feature_cols) - set(features_df.columns)
    if missing_cols:
        logger.warning(f"⚠️ Features faltantes: {missing_cols}")
        # Imputar con 0 (o mediana si disponible)
        for col in missing_cols:
            features_df[col] = 0

    # Ordenar columnas según el modelo
    features_df = features_df[['cell_id', 'lat', 'lon', 'cluster'] + feature_cols]

    logger.info(f"✅ Features preparadas: {len(features_df):,} celdas")
    logger.info(f"   Columnas totales: {len(features_df.columns)}")

    return features_df


def make_predictions(features_df, model, feature_cols, species):
    """
    Genera predicciones de probabilidad

    Parameters:
    -----------
    features_df : pd.DataFrame
        Features preparadas
    model : XGBoost model
        Modelo entrenado
    feature_cols : list
        Columnas de features
    species : str
        Nombre de la especie

    Returns:
    --------
    pd.DataFrame : Predicciones con columnas [cell_id, lat, lon, cluster, probability, species]
    """
    logger.info("\n" + "="*70)
    logger.info("🔮 PASO 4: Generando Predicciones")
    logger.info("="*70)

    # Extraer X
    X = features_df[feature_cols].copy()

    # Validar NaNs
    nan_count = X.isnull().sum().sum()
    if nan_count > 0:
        logger.warning(f"⚠️ {nan_count} NaN detectados. Imputando con mediana...")
        X = X.fillna(X.median())

    # Predecir probabilidades
    logger.info(f"Prediciendo para {len(X):,} celdas...")
    probabilities = model.predict_proba(X)[:, 1]  # Clase 1 (presencia)

    # Crear DataFrame de resultados
    predictions_df = pd.DataFrame({
        'cell_id': features_df['cell_id'],
        'lat': features_df['lat'],
        'lon': features_df['lon'],
        'cluster': features_df['cluster'],
        'probability': probabilities,
        'species': species
    })

    # Estadísticas
    logger.info(f"\n📊 Estadísticas de Predicción:")
    logger.info(f"   Media: {probabilities.mean():.4f}")
    logger.info(f"   Mediana: {np.median(probabilities):.4f}")
    logger.info(f"   Min: {probabilities.min():.4f}")
    logger.info(f"   Max: {probabilities.max():.4f}")

    # Top 10 celdas
    top_10 = predictions_df.nlargest(10, 'probability')
    logger.info(f"\n🏆 Top 10 Celdas con Mayor Probabilidad:")
    for i, row in top_10.iterrows():
        logger.info(f"   {row['probability']:.4f} - Cluster {row['cluster']} "
                   f"({row['lat']:.4f}, {row['lon']:.4f})")

    # Distribución por clusters
    high_prob_cells = predictions_df[predictions_df['probability'] > 0.5]
    if len(high_prob_cells) > 0:
        cluster_dist = high_prob_cells['cluster'].value_counts()
        logger.info(f"\n🎯 Celdas con P > 0.5: {len(high_prob_cells):,}")
        logger.info(f"   Clusters dominantes:")
        for cluster_id, count in cluster_dist.head(5).items():
            pct = count / len(high_prob_cells) * 100
            logger.info(f"      Cluster {cluster_id}: {count:,} celdas ({pct:.1f}%)")
    else:
        logger.info(f"\n⚠️ No hay celdas con P > 0.5")

    logger.info(f"\n✅ Predicciones completadas: {len(predictions_df):,} celdas")

    return predictions_df


def save_predictions(predictions_df, species, target_date, output_dir='outputs/predictions'):
    """
    Guarda predicciones en CSV

    Parameters:
    -----------
    predictions_df : pd.DataFrame
        Predicciones
    species : str
        Nombre de la especie
    target_date : datetime
        Fecha de predicción
    output_dir : str
        Directorio de salida
    """
    logger.info("\n" + "="*70)
    logger.info("💾 PASO 5: Guardando Predicciones")
    logger.info("="*70)

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    # Nombre de archivo
    species_name = species.replace(' ', '_')
    date_str = target_date.strftime('%Y%m%d')
    output_file = output_path / f"{species_name}_{date_str}.csv"

    # Guardar
    predictions_df.to_csv(output_file, index=False)
    logger.info(f"✅ Predicciones guardadas: {output_file}")
    logger.info(f"   Tamaño: {output_file.stat().st_size / 1024 / 1024:.2f} MB")

    # Guardar versión filtrada (P > 0.3)
    high_prob = predictions_df[predictions_df['probability'] > 0.3]
    if len(high_prob) > 0:
        filtered_file = output_path / f"{species_name}_{date_str}_high_prob.csv"
        high_prob.to_csv(filtered_file, index=False)
        logger.info(f"✅ Celdas con P > 0.3: {filtered_file}")
        logger.info(f"   Celdas: {len(high_prob):,}")

    return output_file


def main():
    """
    Pipeline principal de predicción
    """
    parser = argparse.ArgumentParser(description='BuscaFungi - Predicción v2')
    parser.add_argument('--species', type=str, required=True,
                       help='Especie a predecir (ej: "Boletus edulis")')
    parser.add_argument('--date', type=str, default=None,
                       help='Fecha objetivo (YYYY-MM-DD). Default: hoy')
    parser.add_argument('--use-forecast', action='store_true',
                       help='Usar API forecast para fechas futuras')
    parser.add_argument('--output-dir', type=str, default='outputs/predictions',
                       help='Directorio de salida')

    args = parser.parse_args()

    # Parse fecha
    if args.date:
        target_date = datetime.strptime(args.date, '%Y-%m-%d')
    else:
        target_date = datetime.now()

    print("\n" + "="*70)
    print("🍄 BuscaFungi - Predicción v2")
    print("="*70)
    print(f"\nEspecie: {args.species}")
    print(f"Fecha objetivo: {target_date.date()}")
    print(f"Modo: {'Forecast' if args.use_forecast else 'Historical'}")
    print("\n" + "="*70)

    try:
        # Pipeline
        grid_df, model, cluster_features, feature_cols = load_grid_and_models(args.species)

        grid_with_meteo = get_meteorological_data_for_grid(
            grid_df,
            target_date,
            use_forecast=args.use_forecast
        )

        features_df = prepare_prediction_features(
            grid_with_meteo,
            cluster_features,
            feature_cols
        )

        predictions_df = make_predictions(
            features_df,
            model,
            feature_cols,
            args.species
        )

        output_file = save_predictions(
            predictions_df,
            args.species,
            target_date,
            output_dir=args.output_dir
        )

        # Resumen final
        print("\n" + "="*70)
        print("✅ PREDICCIÓN COMPLETADA")
        print("="*70)
        print(f"\n📊 Resumen:")
        print(f"   Especie: {args.species}")
        print(f"   Fecha: {target_date.date()}")
        print(f"   Celdas procesadas: {len(predictions_df):,}")
        print(f"   Probabilidad media: {predictions_df['probability'].mean():.4f}")
        print(f"\n📁 Archivos generados:")
        print(f"   {output_file}")
        print(f"\n💡 Siguiente paso:")
        print(f"   Visualiza las predicciones en un mapa")
        print(f"   O prueba con otra fecha usando --date YYYY-MM-DD")
        print("="*70)

        return 0

    except Exception as e:
        logger.error(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
