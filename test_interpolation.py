#!/usr/bin/env python3
"""
Test de Interpolación Espacial
Valida que la optimización funciona correctamente
"""

import sys
sys.path.insert(0, 'src')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

from src.grid import GridManager
from src.meteo import MeteoDataFetcher

def test_interpolation():
    """
    Test de interpolación espacial para datos meteorológicos
    """
    print("="*70)
    print("🧪 TEST: Interpolación Espacial de Datos Meteorológicos")
    print("="*70)

    # Crear grid pequeño para test
    print("\n1. Creando grid de test (Galicia, 100 celdas)...")

    test_bounds = {
        'lat_min': 42.0,
        'lat_max': 43.0,
        'lon_min': -9.0,
        'lon_max': -7.0
    }

    # Grid manual pequeño
    lats = np.linspace(test_bounds['lat_min'], test_bounds['lat_max'], 10)
    lons = np.linspace(test_bounds['lon_min'], test_bounds['lon_max'], 10)

    lat_grid, lon_grid = np.meshgrid(lats, lons)

    grid_df = pd.DataFrame({
        'cell_id': [f"{lat:.4f}_{lon:.4f}" for lat, lon in zip(lat_grid.flatten(), lon_grid.flatten())],
        'lat': lat_grid.flatten(),
        'lon': lon_grid.flatten()
    })

    print(f"   ✅ Grid creado: {len(grid_df)} celdas")
    print(f"   Área: {test_bounds['lat_min']:.2f}°N - {test_bounds['lat_max']:.2f}°N, "
          f"{test_bounds['lon_min']:.2f}°E - {test_bounds['lon_max']:.2f}°E")

    # Test interpolación
    print("\n2. Testeando interpolación meteorológica...")

    meteo_fetcher = MeteoDataFetcher(enable_disk_cache=True)

    # IMPORTANTE: API archive tiene delay de ~5-7 días
    # Usar fecha antigua para asegurar disponibilidad
    target_date = datetime.now() - timedelta(days=30)  # Hace 30 días (datos disponibles)

    print(f"   Fecha objetivo: {target_date.date()}")
    print(f"   (Nota: API archive tiene delay de ~5 días, por eso usamos fecha antigua)")

    start_time = time.time()

    try:
        grid_with_meteo = meteo_fetcher.get_weather_for_grid(
            grid_df,
            target_date=target_date,
            use_forecast=False,
            sample_resolution_deg=0.5  # ~50km
        )

        elapsed = time.time() - start_time

        if grid_with_meteo is None:
            print("\n   ❌ ERROR: No se pudo obtener datos meteorológicos")
            return False

        print(f"\n   ✅ Interpolación completada en {elapsed:.1f} segundos")

        # Validación
        print("\n3. Validando resultados...")

        meteo_cols = [col for col in grid_with_meteo.columns
                      if col not in ['cell_id', 'lat', 'lon']]

        print(f"   Features meteorológicas: {len(meteo_cols)}")
        print(f"   Columnas: {', '.join(meteo_cols[:5])}...")

        # Check NaNs
        nan_count = grid_with_meteo[meteo_cols].isnull().sum().sum()
        print(f"   NaN detectados: {nan_count}")

        if nan_count > 0:
            print("   ⚠️ Advertencia: Hay valores NaN")

        # Stats de ejemplo
        if 'precip_sum_7d' in grid_with_meteo.columns:
            precip_7d = grid_with_meteo['precip_sum_7d']
            print(f"\n   📊 Precipitación 7d:")
            print(f"      Media: {precip_7d.mean():.1f} mm")
            print(f"      Rango: {precip_7d.min():.1f} - {precip_7d.max():.1f} mm")

        if 'temp_mean_7d' in grid_with_meteo.columns:
            temp_7d = grid_with_meteo['temp_mean_7d']
            print(f"\n   🌡️ Temperatura 7d:")
            print(f"      Media: {temp_7d.mean():.1f} °C")
            print(f"      Rango: {temp_7d.min():.1f} - {temp_7d.max():.1f} °C")

        # Estimación de tiempo para grid completo
        print("\n4. Estimación para grid completo...")

        cells_per_second = len(grid_df) / elapsed

        for grid_size in [1000, 10000, 100000, 500000, 900000]:
            estimated_time = grid_size / cells_per_second
            print(f"   {grid_size:>7,} celdas → ~{estimated_time/60:.1f} minutos")

        print("\n" + "="*70)
        print("✅ TEST COMPLETADO EXITOSAMENTE")
        print("="*70)

        # Guardar ejemplo
        output_file = 'test_interpolation_output.csv'
        grid_with_meteo.to_csv(output_file, index=False)
        print(f"\n💾 Resultado guardado: {output_file}")

        return True

    except Exception as e:
        print(f"\n❌ ERROR durante test: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_interpolation()
    sys.exit(0 if success else 1)
