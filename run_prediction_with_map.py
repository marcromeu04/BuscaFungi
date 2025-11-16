#!/usr/bin/env python3
"""
BuscaFungi - Pipeline Completo con Visualización
Ejecuta predicción y abre mapa automáticamente

Uso:
    python run_prediction_with_map.py --species "Boletus edulis" --date 2024-09-15
    python run_prediction_with_map.py --species "Lactarius deliciosus"  # fecha = hoy
"""

import sys
import subprocess
from pathlib import Path
import argparse
from datetime import datetime


def run_command(cmd, description):
    """Ejecuta comando y muestra progreso"""
    print(f"\n{'='*70}")
    print(f"🚀 {description}")
    print(f"{'='*70}")
    print(f"Comando: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, capture_output=False)

    if result.returncode != 0:
        print(f"\n❌ Error en: {description}")
        return False

    return True


def main():
    parser = argparse.ArgumentParser(description='BuscaFungi - Pipeline Completo')
    parser.add_argument('--species', type=str, required=True,
                       help='Especie (ej: "Boletus edulis")')
    parser.add_argument('--date', type=str, default=None,
                       help='Fecha (YYYY-MM-DD). Default: hoy')
    parser.add_argument('--use-forecast', action='store_true',
                       help='Usar forecast para fechas futuras')
    parser.add_argument('--skip-prediction', action='store_true',
                       help='Saltar predicción (solo visualizar predicción existente)')

    args = parser.parse_args()

    # Fecha
    if args.date:
        target_date = args.date
    else:
        target_date = datetime.now().strftime('%Y-%m-%d')

    print("\n" + "="*70)
    print("🍄 BuscaFungi - Pipeline Completo con Visualización")
    print("="*70)
    print(f"\nEspecie: {args.species}")
    print(f"Fecha: {target_date}")
    print(f"Modo: {'Forecast' if args.use_forecast else 'Historical'}")

    # Paso 1: Verificar que setup y training están completos
    grid_file = Path('outputs/grid_clustered.parquet')
    model_file = Path(f'outputs/models/{args.species.replace(" ", "_")}_v2.joblib')

    if not grid_file.exists():
        print("\n⚠️ Grid no encontrado. Ejecuta primero:")
        print("   python setup_grid_clustering.py")
        return 1

    if not model_file.exists():
        print("\n⚠️ Modelo no encontrado. Ejecuta primero:")
        print("   python train_v2.py")
        return 1

    # Paso 2: Predicción (si no se salta)
    if not args.skip_prediction:
        predict_cmd = [
            sys.executable,
            'predict_v2.py',
            '--species', args.species,
            '--date', target_date
        ]

        if args.use_forecast:
            predict_cmd.append('--use-forecast')

        if not run_command(predict_cmd, f"Predicción para {args.species} en {target_date}"):
            return 1

    # Paso 3: Visualización
    visualize_cmd = [
        sys.executable,
        'visualize_predictions.py',
        '--species', args.species,
        '--date', target_date
    ]

    if not run_command(visualize_cmd, "Generando mapa interactivo"):
        return 1

    # Resumen final
    print("\n" + "="*70)
    print("✅ PIPELINE COMPLETADO")
    print("="*70)
    print(f"\n🎉 El mapa interactivo debería abrirse en tu navegador")
    print(f"\n📁 Archivos generados:")

    # Listar archivos generados
    date_str = target_date.replace('-', '')
    species_name = args.species.replace(' ', '_')

    files = [
        f"outputs/predictions/{species_name}_{date_str}.csv",
        f"outputs/predictions/{species_name}_{date_str}_high_prob.csv",
        "predictions_map.html"
    ]

    for f in files:
        if Path(f).exists():
            print(f"   ✅ {f}")

    print("\n💡 Próximos pasos:")
    print(f"   - Explora el mapa interactivo")
    print(f"   - Prueba con otra fecha: --date YYYY-MM-DD")
    print(f"   - Prueba con otra especie: --species \"Lactarius deliciosus\"")

    print("="*70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
