#!/usr/bin/env python3
"""
BuscaFungi - Training Pipeline v2.0
Entrena modelos usando grid clustered pre-procesado

Requisito: Ejecutar setup_grid_clustering.py primero
"""

import sys
sys.path.insert(0, 'src')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import logging
from pygbif import occurrences as occ, species as gbif_species

from src import config
from src.meteo import MeteoDataFetcher
from src.sdm import MushroomSDM
from src.pseudo_absences import generate_smart_pseudo_absences

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_grid_clustered():
    """
    Carga grid pre-procesado desde setup_grid_clustering.py
    """
    logger.info("\n📥 Cargando grid clustered...")

    grid_file = Path('outputs/grid_clustered.parquet')

    if not grid_file.exists():
        logger.error(f"❌ No existe: {grid_file}")
        logger.error(f"   Ejecuta primero: python setup_grid_clustering.py")
        sys.exit(1)

    grid_df = pd.read_parquet(grid_file)

    logger.info(f"✅ Grid cargado: {len(grid_df):,} celdas")
    logger.info(f"   Columnas: {len(grid_df.columns)}")
    logger.info(f"   Clusters: {grid_df['cluster'].nunique()}")

    return grid_df


def fetch_gbif_observations(species_name, bounds, limit=500):
    """
    Descarga observaciones de GBIF para una especie
    """
    logger.info(f"\n🔍 Buscando: {species_name}")

    try:
        # Obtener taxon key
        result = gbif_species.name_backbone(name=species_name)
        if 'usageKey' not in result:
            logger.warning(f"  ❌ No encontrado en GBIF")
            return None

        taxon_key = result['usageKey']
        logger.info(f"  ✅ GBIF key: {taxon_key}")

        # Buscar observaciones
        results = occ.search(
            taxonKey=taxon_key,
            country='ES',
            hasCoordinate=True,
            hasGeospatialIssue=False,
            limit=limit,
            year='2015,2024'
        )

        count = results.get('count', 0)

        if count == 0:
            logger.warning(f"  ⚠️ 0 observaciones")
            return None

        # Parsear observaciones
        obs_list = []
        for obs in results.get('results', []):
            if 'decimalLatitude' in obs and 'decimalLongitude' in obs:
                lat = obs['decimalLatitude']
                lon = obs['decimalLongitude']

                # Filtrar por región de interés
                if (bounds['lat_min'] <= lat <= bounds['lat_max'] and
                    bounds['lon_min'] <= lon <= bounds['lon_max']):

                    # Fecha
                    date = None
                    if 'eventDate' in obs:
                        try:
                            date = pd.to_datetime(obs['eventDate'])
                        except:
                            pass

                    if date is None and 'year' in obs and 'month' in obs:
                        year = obs['year']
                        month = obs.get('month', 1)
                        day = obs.get('day', 1)
                        try:
                            date = datetime(year, month, day)
                        except:
                            continue

                    if date is None:
                        continue

                    obs_list.append({
                        'species': species_name,
                        'lat': lat,
                        'lon': lon,
                        'date': date,
                        'observed': 1
                    })

        if obs_list:
            df = pd.DataFrame(obs_list)
            logger.info(f"  ✅ {len(df)} observaciones válidas")
            return df
        else:
            logger.warning(f"  ⚠️ 0 observaciones válidas")
            return None

    except Exception as e:
        logger.error(f"  ❌ Error: {e}")
        return None


def assign_observations_to_grid(observations_df, grid_df):
    """
    Asigna observaciones a celdas del grid (y sus clusters)

    CLAVE: Cada observación hereda el cluster de su celda
    """
    logger.info(f"\n📍 Asignando {len(observations_df)} observaciones al grid...")

    from src.utils import haversine_distance

    obs_with_cluster = []

    for idx, obs in observations_df.iterrows():
        # Encontrar celda más cercana
        distances = np.sqrt(
            (grid_df['lat'] - obs['lat'])**2 +
            (grid_df['lon'] - obs['lon'])**2
        )

        nearest_idx = distances.idxmin()
        nearest_cell = grid_df.iloc[nearest_idx]

        # Distancia real
        dist_km = haversine_distance(
            obs['lat'], obs['lon'],
            nearest_cell['lat'], nearest_cell['lon']
        )

        if dist_km > 5.0:  # Tolerancia 5km
            logger.warning(f"  ⚠️ Obs muy lejos de grid: {dist_km:.1f}km")

        # Asignar cluster de la celda
        obs_dict = obs.to_dict()
        obs_dict['cell_id'] = nearest_cell['cell_id']
        obs_dict['cluster'] = nearest_cell['cluster']
        obs_dict['snap_distance_km'] = dist_km

        obs_with_cluster.append(obs_dict)

    result_df = pd.DataFrame(obs_with_cluster)

    logger.info(f"  ✅ {len(result_df)} obs asignadas")
    logger.info(f"  📊 Clusters con presencias: {result_df['cluster'].nunique()}")

    return result_df


def add_meteorological_features_to_observations(observations_df):
    """
    Añade features meteorológicas de los 30 días ANTES de cada observación

    CLAVE: Solo para observaciones (no todo el grid)
    """
    logger.info(f"\n🌧️ Añadiendo features meteorológicas a observaciones...")

    meteo_fetcher = MeteoDataFetcher(enable_disk_cache=True)

    # Agrupar por ubicación para minimizar API calls
    n_locations = observations_df.groupby(['lat', 'lon']).ngroups

    logger.info(f"  📥 Observaciones: {len(observations_df)}")
    logger.info(f"  📍 Ubicaciones únicas: {n_locations}")
    logger.info(f"  ⏱️ Tiempo estimado: ~{n_locations * 2} segundos")

    obs_with_meteo = []

    grouped = observations_df.groupby(['lat', 'lon'])

    for (lat, lon), group in grouped:
        # Rango de fechas para esta ubicación
        min_date = group['date'].min() - timedelta(days=30)
        max_date = group['date'].max()

        # Fetch weather
        weather_df = meteo_fetcher.fetch_historical_weather(lat, lon, min_date, max_date)

        if weather_df is None:
            logger.warning(f"  ⚠️ No meteo para ({lat:.4f}, {lon:.4f})")
            continue

        # Para cada observación en esta ubicación
        for idx, obs in group.iterrows():
            obs_date = obs['date']

            # Calcular features temporales (30 días antes)
            meteo_features = meteo_fetcher.calculate_temporal_features(weather_df, obs_date)

            # Combinar
            obs_dict = obs.to_dict()
            obs_dict.update(meteo_features)

            obs_with_meteo.append(obs_dict)

        logger.info(f"  ✅ Procesadas {len(group)} obs en ({lat:.4f}, {lon:.4f})")

    result_df = pd.DataFrame(obs_with_meteo)
    logger.info(f"\n✅ {len(result_df)} observaciones con meteo")

    return result_df


def extract_cluster_features(grid_df):
    """
    Extrae features PROMEDIO por cluster

    En lugar de features por celda, usamos features por cluster
    """
    logger.info(f"\n📊 Calculando features promedio por cluster...")

    # Features ambientales a agregar
    feature_cols = [
        'ph', 'clay_percent', 'sand_percent', 'organic_carbon',
        'elevation', 'slope', 'aspect_sin', 'aspect_cos', 'twi'
    ]

    # Añadir veg dummies
    veg_cols = [col for col in grid_df.columns if col.startswith('veg_')]
    feature_cols.extend(veg_cols)

    # Filtrar solo existentes
    feature_cols = [col for col in feature_cols if col in grid_df.columns]

    cluster_features = {}

    for cluster_id in sorted(grid_df['cluster'].unique()):
        cluster_cells = grid_df[grid_df['cluster'] == cluster_id]

        # Promedio de features numéricas
        cluster_stats = {}
        for col in feature_cols:
            cluster_stats[col] = cluster_cells[col].mean()

        cluster_features[cluster_id] = cluster_stats

    logger.info(f"  ✅ {len(cluster_features)} clusters procesados")

    return cluster_features, feature_cols


def prepare_training_data(observations_df, cluster_features, feature_cols):
    """
    Prepara datos de entrenamiento usando cluster features + meteo

    Features finales = cluster_features[obs.cluster] + meteo_features
    """
    logger.info(f"\n🔧 Preparando datos de entrenamiento...")

    training_data = []

    for idx, obs in observations_df.iterrows():
        cluster_id = obs['cluster']

        # Features del cluster
        obs_features = cluster_features[cluster_id].copy()

        # Añadir features meteorológicas
        meteo_cols = [col for col in obs.index if col.startswith(('precip_', 'temp_', 'rain_', 'days_since', 'sunshine_', 'soil_moisture'))]

        for col in meteo_cols:
            obs_features[col] = obs[col]

        # Metadata
        obs_features['species'] = obs['species']
        obs_features['cluster'] = cluster_id
        obs_features['observed'] = 1

        training_data.append(obs_features)

    training_df = pd.DataFrame(training_data)

    logger.info(f"  ✅ {len(training_df)} presencias preparadas")
    logger.info(f"  📊 Features totales: {len(training_df.columns) - 3}")  # -3 para metadata

    return training_df


def train_models(presences_df, cluster_features, feature_cols, grid_df):
    """
    Entrena modelos SDM por especie
    """
    logger.info(f"\n🤖 Entrenando modelos...")

    models = {}

    for species in presences_df['species'].unique():
        logger.info(f"\n{'='*60}")
        logger.info(f"Especie: {species}")
        logger.info(f"{'='*60}")

        # Presencias de esta especie
        species_presences = presences_df[presences_df['species'] == species].copy()

        if len(species_presences) < 20:
            logger.warning(f"  ⚠️ Solo {len(species_presences)} obs. Mínimo: 20")
            continue

        logger.info(f"  ✅ {len(species_presences)} presencias")

        # Generar pseudo-ausencias
        logger.info(f"\n  🎲 Generando pseudo-ausencias...")

        # Para pseudo-ausencias, necesitamos celdas del grid
        # Usamos clusters DIFERENTES a donde hay presencias
        presence_clusters = set(species_presences['cluster'].unique())

        logger.info(f"     Clusters con presencias: {presence_clusters}")

        # Celdas candidatas para ausencias (otros clusters)
        absence_candidates = grid_df[~grid_df['cluster'].isin(presence_clusters)]

        if len(absence_candidates) < len(species_presences):
            logger.warning(f"     Pocos candidatos. Usando todos los clusters.")
            absence_candidates = grid_df

        # Samplear ausencias
        n_absences = int(len(species_presences) * config.PSEUDO_ABSENCE_RATIO)
        n_absences = min(n_absences, len(absence_candidates))

        absence_cells = absence_candidates.sample(n=n_absences, random_state=42)

        logger.info(f"     ✅ {len(absence_cells)} pseudo-ausencias")

        # Preparar ausencias con cluster features
        absence_data = []

        for idx, cell in absence_cells.iterrows():
            cluster_id = cell['cluster']

            # Features del cluster
            abs_features = cluster_features[cluster_id].copy()

            # Meteo: usar promedio de presencias (aproximación)
            # En producción: generar meteo aleatorio de esa época del año
            meteo_cols = [col for col in species_presences.columns if col.startswith(('precip_', 'temp_', 'rain_', 'days_since', 'sunshine_', 'soil_moisture'))]

            for col in meteo_cols:
                abs_features[col] = species_presences[col].median()

            abs_features['species'] = species
            abs_features['cluster'] = cluster_id
            abs_features['observed'] = 0

            absence_data.append(abs_features)

        absences_df = pd.DataFrame(absence_data)

        # Combinar presencias + ausencias
        combined_df = pd.concat([species_presences, absences_df], ignore_index=True)

        logger.info(f"\n  📊 Dataset final:")
        logger.info(f"     Presencias: {len(species_presences)}")
        logger.info(f"     Ausencias:  {len(absences_df)}")
        logger.info(f"     Total:      {len(combined_df)}")

        # Preparar features para modelo
        X_cols = feature_cols + [col for col in combined_df.columns if col.startswith(('precip_', 'temp_', 'rain_', 'days_since', 'sunshine_', 'soil_moisture'))]
        X_cols = [col for col in X_cols if col in combined_df.columns]

        X = combined_df[X_cols]
        y = combined_df['observed'].values

        # Imputar NaNs
        if X.isnull().any().any():
            logger.warning(f"  ⚠️ NaN detectados. Imputando...")
            X = X.fillna(X.median())

        # Entrenar (simplificado, sin validación espacial por ahora)
        from sklearn.preprocessing import StandardScaler
        import xgboost as xgb

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = xgb.XGBClassifier(**config.XGBOOST_PARAMS)

        logger.info(f"\n  🏗️ Entrenando modelo...")
        model.fit(X_scaled, y)

        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        logger.info(f"\n  🔝 Top 10 features:")
        for idx, row in feature_importance.head(10).iterrows():
            logger.info(f"     {row['feature']:30} {row['importance']:.3f}")

        # Guardar
        models[species] = {
            'model': model,
            'scaler': scaler,
            'feature_cols': X_cols
        }

        logger.info(f"\n  ✅ Modelo entrenado")

    return models


def main():
    """
    Pipeline principal de entrenamiento v2.0
    """
    print("\n" + "="*70)
    print("🍄 BuscaFungi - Training Pipeline v2.0")
    print("="*70)
    print(f"\nRegión: {config.FOCUS_REGION}")
    print(f"Especies: {len(config.SPECIES_CONFIG)}")
    print("\n" + "="*70)

    # ========== PASO 1: CARGAR GRID ==========
    grid_df = load_grid_clustered()

    # ========== PASO 2: DESCARGAR GBIF ==========
    logger.info("\n" + "="*70)
    logger.info("PASO 2: Descargar Observaciones GBIF")
    logger.info("="*70)

    all_observations = []

    for species_name in config.SPECIES_CONFIG.keys():
        obs = fetch_gbif_observations(species_name, config.SPAIN_BOUNDS, limit=500)

        if obs is not None:
            all_observations.append(obs)

    if len(all_observations) == 0:
        logger.error("\n❌ No se pudieron descargar observaciones")
        sys.exit(1)

    observations_df = pd.concat(all_observations, ignore_index=True)

    logger.info(f"\n📊 Total observaciones: {len(observations_df)}")
    logger.info("\nDistribución por especie:")
    print(observations_df['species'].value_counts())

    # ========== PASO 3: ASIGNAR A GRID ==========
    logger.info("\n" + "="*70)
    logger.info("PASO 3: Asignar Observaciones al Grid")
    logger.info("="*70)

    observations_df = assign_observations_to_grid(observations_df, grid_df)

    # ========== PASO 4: AÑADIR METEO ==========
    logger.info("\n" + "="*70)
    logger.info("PASO 4: Añadir Features Meteorológicas (30 días)")
    logger.info("="*70)

    observations_df = add_meteorological_features_to_observations(observations_df)

    # ========== PASO 5: CLUSTER FEATURES ==========
    logger.info("\n" + "="*70)
    logger.info("PASO 5: Calcular Cluster Features")
    logger.info("="*70)

    cluster_features, feature_cols = extract_cluster_features(grid_df)

    # ========== PASO 6: PREPARAR DATOS ==========
    logger.info("\n" + "="*70)
    logger.info("PASO 6: Preparar Datos de Entrenamiento")
    logger.info("="*70)

    training_df = prepare_training_data(observations_df, cluster_features, feature_cols)

    # ========== PASO 7: ENTRENAR ==========
    logger.info("\n" + "="*70)
    logger.info("PASO 7: Entrenar Modelos")
    logger.info("="*70)

    models = train_models(training_df, cluster_features, feature_cols, grid_df)

    # ========== PASO 8: GUARDAR ==========
    logger.info("\n" + "="*70)
    logger.info("PASO 8: Guardar Modelos")
    logger.info("="*70)

    import joblib

    output_dir = Path('outputs/models')
    output_dir.mkdir(exist_ok=True, parents=True)

    for species, model_data in models.items():
        model_file = output_dir / f"{species.replace(' ', '_')}_v2.joblib"
        joblib.dump(model_data, model_file)
        logger.info(f"✅ {species}: {model_file}")

    # Guardar también cluster features
    cluster_file = Path('outputs/cluster_features.joblib')
    joblib.dump({
        'cluster_features': cluster_features,
        'feature_cols': feature_cols
    }, cluster_file)
    logger.info(f"✅ Cluster features: {cluster_file}")

    # ========== RESUMEN ==========
    print("\n" + "="*70)
    print("✅ ENTRENAMIENTO COMPLETADO")
    print("="*70)
    print(f"\n📊 Resumen:")
    print(f"   Modelos entrenados: {len(models)}")
    print(f"   Observaciones: {len(observations_df)}")
    print(f"   Clusters: {len(cluster_features)}")
    print(f"\n📁 Archivos generados:")
    print(f"   - outputs/models/*_v2.joblib")
    print(f"   - outputs/cluster_features.joblib")
    print(f"\n🎯 Siguiente paso:")
    print(f"   python predict_v2.py --date 2024-11-15 --species 'Boletus edulis'")
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
