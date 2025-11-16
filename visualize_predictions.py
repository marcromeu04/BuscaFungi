#!/usr/bin/env python3
"""
BuscaFungi - Visualización Interactiva de Predicciones
Genera mapa interactivo con probabilidades de hongos

Uso:
    python visualize_predictions.py --file outputs/predictions/Boletus_edulis_20240915.csv
    python visualize_predictions.py --species "Boletus edulis" --date 2024-09-15
"""

import sys
sys.path.insert(0, 'src')

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import argparse
import logging
import webbrowser

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_interactive_map(predictions_df, species, date, output_file='map.html'):
    """
    Crea mapa interactivo con Folium

    Parameters:
    -----------
    predictions_df : pd.DataFrame
        Predicciones con columnas [lat, lon, probability]
    species : str
        Nombre de la especie
    date : str
        Fecha de predicción
    output_file : str
        Archivo HTML de salida
    """
    try:
        import folium
        from folium import plugins
    except ImportError:
        logger.error("❌ Folium no instalado. Instalando...")
        import subprocess
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'folium'])
        import folium
        from folium import plugins

    logger.info(f"\n🗺️ Creando mapa interactivo para {species}")
    logger.info(f"   Fecha: {date}")
    logger.info(f"   Celdas: {len(predictions_df):,}")

    # Filtrar solo probabilidades significativas (> 0.1) para mejor rendimiento
    significant = predictions_df[predictions_df['probability'] > 0.1].copy()
    logger.info(f"   Celdas con P > 0.1: {len(significant):,}")

    if len(significant) == 0:
        logger.warning("⚠️ No hay celdas con probabilidad > 0.1")
        significant = predictions_df.nlargest(100, 'probability')
        logger.info(f"   Mostrando top 100 celdas")

    # Centro del mapa (promedio ponderado por probabilidad)
    center_lat = (significant['lat'] * significant['probability']).sum() / significant['probability'].sum()
    center_lon = (significant['lon'] * significant['probability']).sum() / significant['probability'].sum()

    # Crear mapa base
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=7,
        tiles='OpenStreetMap',
        control_scale=True
    )

    # Añadir capas base alternativas
    folium.TileLayer('Stamen Terrain', name='Terreno').add_to(m)
    folium.TileLayer('CartoDB positron', name='Claro').add_to(m)

    # Preparar datos para heatmap
    heat_data = []
    for _, row in significant.iterrows():
        # [lat, lon, weight]
        heat_data.append([row['lat'], row['lon'], row['probability']])

    # Añadir heatmap
    plugins.HeatMap(
        heat_data,
        name='Mapa de Calor',
        min_opacity=0.3,
        max_opacity=0.8,
        radius=15,
        blur=20,
        gradient={
            0.0: 'blue',
            0.3: 'lime',
            0.5: 'yellow',
            0.7: 'orange',
            1.0: 'red'
        }
    ).add_to(m)

    # Añadir marcadores para top 20 celdas
    top_20 = significant.nlargest(20, 'probability')

    feature_group = folium.FeatureGroup(name='Top 20 Ubicaciones')

    for idx, row in top_20.iterrows():
        # Color según probabilidad
        if row['probability'] >= 0.7:
            color = 'darkgreen'
            icon = 'star'
        elif row['probability'] >= 0.5:
            color = 'green'
            icon = 'leaf'
        elif row['probability'] >= 0.3:
            color = 'orange'
            icon = 'circle'
        else:
            color = 'lightgray'
            icon = 'circle'

        # Popup con información
        popup_html = f"""
        <div style="font-family: Arial; width: 200px;">
            <h4 style="margin: 0 0 10px 0;">{species}</h4>
            <hr style="margin: 5px 0;">
            <b>📍 Ubicación:</b><br>
            Lat: {row['lat']:.4f}°<br>
            Lon: {row['lon']:.4f}°<br>
            <br>
            <b>🎯 Probabilidad:</b><br>
            <span style="font-size: 18px; color: {color};">
                {row['probability']*100:.1f}%
            </span><br>
            <br>
            <b>🌳 Cluster:</b> {int(row['cluster'])}<br>
            <br>
            <small style="color: gray;">
            Fecha: {date}
            </small>
        </div>
        """

        folium.Marker(
            location=[row['lat'], row['lon']],
            popup=folium.Popup(popup_html, max_width=250),
            tooltip=f"P={row['probability']*100:.1f}%",
            icon=folium.Icon(color=color, icon=icon, prefix='fa')
        ).add_to(feature_group)

    feature_group.add_to(m)

    # Añadir leyenda
    legend_html = f'''
    <div style="
        position: fixed;
        bottom: 50px;
        left: 50px;
        width: 250px;
        background-color: white;
        border: 2px solid grey;
        z-index: 9999;
        font-size: 14px;
        padding: 10px;
        border-radius: 5px;
        box-shadow: 0 0 15px rgba(0,0,0,0.2);
    ">
        <h4 style="margin: 0 0 10px 0;">🍄 {species}</h4>
        <p style="margin: 5px 0;"><b>Fecha:</b> {date}</p>
        <p style="margin: 5px 0;"><b>Celdas analizadas:</b> {len(predictions_df):,}</p>
        <hr>
        <p style="margin: 5px 0;"><b>Probabilidad:</b></p>
        <div style="margin: 5px 0;">
            <span style="color: darkgreen;">⬤</span> Muy Alta (>70%)<br>
            <span style="color: green;">⬤</span> Alta (50-70%)<br>
            <span style="color: orange;">⬤</span> Media (30-50%)<br>
            <span style="color: lightgray;">⬤</span> Baja (<30%)<br>
        </div>
        <hr>
        <p style="margin: 5px 0; font-size: 12px; color: gray;">
            Max: {predictions_df['probability'].max()*100:.1f}%<br>
            Media: {predictions_df['probability'].mean()*100:.1f}%
        </p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))

    # Control de capas
    folium.LayerControl().add_to(m)

    # Guardar
    m.save(output_file)
    logger.info(f"✅ Mapa guardado: {output_file}")

    return output_file


def load_predictions(file_path=None, species=None, date=None):
    """
    Carga predicciones desde archivo o busca por especie/fecha

    Parameters:
    -----------
    file_path : str, optional
        Ruta directa al archivo CSV
    species : str, optional
        Nombre de la especie
    date : str, optional
        Fecha (YYYY-MM-DD)

    Returns:
    --------
    tuple: (predictions_df, species, date_str)
    """
    if file_path:
        # Cargar desde archivo directo
        predictions_df = pd.read_csv(file_path)

        # Extraer metadatos del nombre del archivo
        filename = Path(file_path).stem
        parts = filename.split('_')

        if len(parts) >= 3:
            species = ' '.join(parts[:-1])  # Todo menos la fecha
            date_str = parts[-1]  # Última parte es la fecha
            # Formatear fecha YYYYMMDD -> YYYY-MM-DD
            if len(date_str) == 8:
                date_str = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
        else:
            species = predictions_df['species'].iloc[0] if 'species' in predictions_df else 'Unknown'
            date_str = 'Unknown'

        logger.info(f"📂 Cargadas predicciones desde: {file_path}")

    elif species and date:
        # Buscar archivo por especie y fecha
        species_name = species.replace(' ', '_')
        date_str = date.replace('-', '')

        search_patterns = [
            f"outputs/predictions/{species_name}_{date_str}.csv",
            f"outputs/predictions/{species_name}*.csv",
        ]

        file_path = None
        for pattern in search_patterns:
            matches = list(Path('.').glob(pattern))
            if matches:
                file_path = matches[0]
                break

        if not file_path:
            raise FileNotFoundError(
                f"No se encontró archivo de predicciones para {species} en {date}\n"
                f"Busca en: outputs/predictions/\n"
                f"O ejecuta primero: python predict_v2.py --species \"{species}\" --date {date}"
            )

        predictions_df = pd.read_csv(file_path)
        date_str = date
        logger.info(f"📂 Encontrado: {file_path}")

    else:
        raise ValueError("Debes proporcionar --file O (--species + --date)")

    logger.info(f"   Especie: {species}")
    logger.info(f"   Fecha: {date_str}")
    logger.info(f"   Registros: {len(predictions_df):,}")

    return predictions_df, species, date_str


def main():
    """
    Pipeline principal de visualización
    """
    parser = argparse.ArgumentParser(description='BuscaFungi - Visualización de Predicciones')

    # Opción 1: Archivo directo
    parser.add_argument('--file', type=str, help='Archivo CSV de predicciones')

    # Opción 2: Por especie y fecha
    parser.add_argument('--species', type=str, help='Especie (ej: "Boletus edulis")')
    parser.add_argument('--date', type=str, help='Fecha (YYYY-MM-DD)')

    # Output
    parser.add_argument('--output', type=str, default='predictions_map.html',
                       help='Archivo HTML de salida (default: predictions_map.html)')
    parser.add_argument('--no-open', action='store_true',
                       help='No abrir automáticamente el navegador')

    args = parser.parse_args()

    print("\n" + "="*70)
    print("🗺️ BuscaFungi - Visualización Interactiva")
    print("="*70)

    try:
        # Cargar predicciones
        predictions_df, species, date_str = load_predictions(
            file_path=args.file,
            species=args.species,
            date=args.date
        )

        # Crear mapa
        output_file = create_interactive_map(
            predictions_df,
            species,
            date_str,
            output_file=args.output
        )

        # Abrir en navegador
        if not args.no_open:
            logger.info(f"\n🌐 Abriendo mapa en navegador...")
            output_path = Path(output_file).absolute()
            webbrowser.open(f'file://{output_path}')

        # Resumen
        print("\n" + "="*70)
        print("✅ VISUALIZACIÓN COMPLETADA")
        print("="*70)
        print(f"\n📊 Resumen:")
        print(f"   Especie: {species}")
        print(f"   Fecha: {date_str}")
        print(f"   Celdas totales: {len(predictions_df):,}")
        print(f"   Probabilidad máxima: {predictions_df['probability'].max()*100:.1f}%")
        print(f"   Probabilidad media: {predictions_df['probability'].mean()*100:.1f}%")

        high_prob = predictions_df[predictions_df['probability'] > 0.5]
        if len(high_prob) > 0:
            print(f"\n🎯 Zonas con alta probabilidad (>50%):")
            print(f"   {len(high_prob):,} celdas")
            print(f"   Clusters: {sorted(high_prob['cluster'].unique().tolist())}")

        print(f"\n📁 Mapa guardado en: {output_file}")
        if not args.no_open:
            print(f"   (Debería abrirse automáticamente en tu navegador)")

        print("="*70)

        return 0

    except Exception as e:
        logger.error(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
