#!/usr/bin/env python3
"""
BuscaFungi - Demo Rápido de Visualización
Genera predicciones de ejemplo y abre el mapa

SOLO PARA DEMOSTRACIÓN - Usa datos sintéticos
Para predicciones reales, usa: run_prediction_with_map.py
"""

import sys
sys.path.insert(0, 'src')

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import webbrowser

print("\n" + "="*70)
print("🍄 BuscaFungi - Demo de Visualización")
print("="*70)
print("\n⚠️ NOTA: Esto genera datos SINTÉTICOS solo para demostración")
print("Para predicciones reales, primero ejecuta:")
print("  1. python setup_grid_clustering.py")
print("  2. python train_v2.py")
print("  3. python run_prediction_with_map.py --species \"Boletus edulis\" --date 2024-09-15")
print("\n" + "="*70)

# Crear datos de ejemplo
print("\n📊 Generando predicciones de ejemplo...")

# Grid de España del norte (Galicia, Asturias, León)
np.random.seed(42)

# Generar 5000 celdas de ejemplo
n_cells = 5000

# Área de ejemplo (norte de España)
lats = np.random.uniform(42.0, 43.8, n_cells)
lons = np.random.uniform(-8.5, -5.0, n_cells)

# Generar probabilidades con patrones realistas
# Zonas de alta probabilidad en ciertas áreas
high_prob_centers = [
    (42.5, -7.0),  # León
    (43.0, -7.5),  # Asturias
    (42.8, -6.5),  # Bierzo
]

probabilities = np.zeros(n_cells)

for i in range(n_cells):
    lat, lon = lats[i], lons[i]

    # Calcular distancia a centros de alta probabilidad
    min_dist = float('inf')
    for center_lat, center_lon in high_prob_centers:
        dist = np.sqrt((lat - center_lat)**2 + (lon - center_lon)**2)
        min_dist = min(min_dist, dist)

    # Probabilidad inversamente proporcional a distancia + ruido
    base_prob = 0.8 * np.exp(-min_dist * 3)
    noise = np.random.uniform(-0.2, 0.2)
    probabilities[i] = np.clip(base_prob + noise, 0, 1)

# Asignar clusters (simulado)
clusters = np.random.randint(0, 15, n_cells)

# Crear DataFrame
predictions_df = pd.DataFrame({
    'cell_id': [f"{lat:.6f}_{lon:.6f}" for lat, lon in zip(lats, lons)],
    'lat': lats,
    'lon': lons,
    'cluster': clusters,
    'probability': probabilities,
    'species': 'Boletus edulis (DEMO)'
})

print(f"✅ Generadas {len(predictions_df):,} celdas de ejemplo")
print(f"   Probabilidad máxima: {probabilities.max()*100:.1f}%")
print(f"   Probabilidad media: {probabilities.mean()*100:.1f}%")

# Guardar temporal
demo_dir = Path('outputs/predictions')
demo_dir.mkdir(exist_ok=True, parents=True)

demo_file = demo_dir / 'DEMO_Boletus_edulis.csv'
predictions_df.to_csv(demo_file, index=False)

print(f"\n💾 Guardado en: {demo_file}")

# Visualizar
print("\n🗺️ Generando mapa interactivo...")

try:
    from visualize_predictions import create_interactive_map

    output_file = 'demo_map.html'
    create_interactive_map(
        predictions_df,
        species='Boletus edulis (DEMO)',
        date=datetime.now().strftime('%Y-%m-%d'),
        output_file=output_file
    )

    # Abrir en navegador
    print(f"\n🌐 Abriendo mapa en navegador...")
    output_path = Path(output_file).absolute()
    webbrowser.open(f'file://{output_path}')

    print("\n" + "="*70)
    print("✅ DEMO COMPLETADO")
    print("="*70)
    print(f"\n📁 Mapa guardado en: {output_file}")
    print(f"   (Debería abrirse automáticamente en tu navegador)")
    print(f"\n💡 Explora el mapa:")
    print(f"   - Zoom con scroll o botones +/-")
    print(f"   - Click en marcadores para ver detalles")
    print(f"   - Cambia capas con el control de capas (arriba derecha)")
    print(f"   - Heatmap muestra densidad de probabilidades")
    print("\n🔥 Para predicciones REALES con datos científicos:")
    print(f"   1. Configura API key: ver API_KEY_SETUP.md")
    print(f"   2. Ejecuta: python setup_grid_clustering.py")
    print(f"   3. Ejecuta: python train_v2.py")
    print(f"   4. Ejecuta: python run_prediction_with_map.py --species \"Boletus edulis\" --date 2024-09-15")
    print("="*70)

except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
