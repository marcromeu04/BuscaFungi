#!/usr/bin/env python3
"""
BuscaFungi - Script de Entrenamiento
Entrena modelos SDM para hongos usando datos de GBIF
"""

import sys
import pandas as pd
from datetime import datetime
from pygbif import occurrences as occ, species as gbif_species

# Añadir src al path
sys.path.insert(0, 'src')

from src.pipeline import BuscaFungiPipeline
from src import config


def fetch_gbif_observations(species_name, bounds, limit=500):
    """
    Descarga observaciones de GBIF para una especie
    """
    print(f"\n🔍 Buscando: {species_name}")

    try:
        # Obtener taxon key
        result = gbif_species.name_backbone(name=species_name)
        if 'usageKey' not in result:
            print(f"  ❌ No encontrado en GBIF")
            return None

        taxon_key = result['usageKey']
        print(f"  ✅ GBIF key: {taxon_key}")

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
            print(f"  ⚠️ 0 observaciones")
            return None

        # Parsear observaciones
        obs_list = []
        for obs in results.get('results', []):
            if 'decimalLatitude' in obs and 'decimalLongitude' in obs:
                lat = obs['decimalLatitude']
                lon = obs['decimalLongitude']

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
            print(f"  ✅ {len(df)} observaciones válidas")
            return df
        else:
            print(f"  ⚠️ 0 observaciones válidas")
            return None

    except Exception as e:
        print(f"  ❌ Error: {e}")
        return None


def main():
    """
    Pipeline principal de entrenamiento
    """
    print("\n" + "="*70)
    print("🍄 BuscaFungi - Entrenamiento de Modelos SDM")
    print("="*70)

    # Configuración
    print(f"\n📍 Región: {config.FOCUS_REGION}")
    print(f"🎯 Resolución: {config.GRID_RESOLUTION_KM}km")
    print(f"⚡ Modo muestra: {config.USE_SAMPLE}")

    # Descargar observaciones de GBIF
    print("\n" + "="*70)
    print("📥 Descargando observaciones de GBIF")
    print("="*70)

    all_observations = []

    for species_name in config.SPECIES_CONFIG.keys():
        obs = fetch_gbif_observations(
            species_name,
            config.SPAIN_BOUNDS,
            limit=500
        )

        if obs is not None:
            all_observations.append(obs)

    if len(all_observations) == 0:
        print("\n❌ No se pudieron descargar observaciones. Abortando.")
        return

    observations_df = pd.concat(all_observations, ignore_index=True)

    print(f"\n📊 Total observaciones: {len(observations_df)}")
    print("\nDistribución por especie:")
    print(observations_df['species'].value_counts())

    # Inicializar pipeline
    pipeline = BuscaFungiPipeline(
        use_sample=config.USE_SAMPLE,
        sample_size=config.SAMPLE_SIZE
    )

    # Ejecutar pipeline completo
    try:
        results = pipeline.run_full_pipeline(observations_df)

        # Guardar resultados
        print("\n" + "="*70)
        print("💾 Guardando resultados...")
        print("="*70)

        pipeline.save_pipeline('outputs')

        print("\n✅ ¡Entrenamiento completado!")
        print("\n📊 Modelos entrenados:")
        for species in pipeline.models:
            print(f"  - {species}")

        print("\n📁 Archivos guardados en: outputs/")
        print("  - grid.csv")
        print("  - features.csv")
        print("  - observations.csv")
        print("  - models/*.joblib")

    except Exception as e:
        print(f"\n❌ Error durante entrenamiento: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
