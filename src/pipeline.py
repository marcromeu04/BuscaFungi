"""
BuscaFungi - Main Pipeline
Pipeline principal de entrenamiento y predicción
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from pathlib import Path

from . import config
from .grid import GridManager
from .features import FeatureExtractor
from .meteo import MeteoDataFetcher
from .clustering import EcologicalClusterer
from .sdm import MushroomSDM
from .utils import print_section_header

# Setup logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT
)
logger = logging.getLogger(__name__)


class BuscaFungiPipeline:
    """
    Pipeline completo para predicción de hongos
    """

    def __init__(self, use_sample=None, sample_size=None):
        """
        Parameters:
        -----------
        use_sample : bool, optional
            Si True, usa muestra del grid (para testing rápido)
        sample_size : int, optional
            Tamaño de muestra
        """
        self.use_sample = use_sample if use_sample is not None else config.USE_SAMPLE
        self.sample_size = sample_size or config.SAMPLE_SIZE

        # Componentes
        self.grid_manager = GridManager()
        self.feature_extractor = FeatureExtractor()
        self.meteo_fetcher = MeteoDataFetcher()
        self.clusterer = EcologicalClusterer()

        # Estado
        self.grid_df = None
        self.features_df = None
        self.observations_df = None
        self.models = {}

        logger.info("="*60)
        logger.info("🍄 BuscaFungi Pipeline Inicializado")
        logger.info("="*60)

    def setup_grid(self):
        """
        Paso 1: Crear grid fijo
        """
        print_section_header("PASO 1: Creando Grid", "🗺️")

        self.grid_df = self.grid_manager.create_grid(
            use_sample=self.use_sample,
            sample_size=self.sample_size
        )

        return self.grid_df

    def load_observations(self, observations_df: pd.DataFrame):
        """
        Paso 2: Cargar observaciones y asignarlas al grid

        Parameters:
        -----------
        observations_df : pd.DataFrame
            Observaciones con columnas ['species', 'lat', 'lon', 'date', ...]
        """
        print_section_header("PASO 2: Cargando Observaciones", "📥")

        # Asignar observaciones a celdas del grid
        self.observations_df = self.grid_manager.assign_observations_to_grid(
            observations_df
        )

        # Resumen
        logger.info("\n📊 Resumen de observaciones:")
        for species in self.observations_df['species'].unique():
            count = len(self.observations_df[self.observations_df['species'] == species])
            logger.info(f"  {species}: {count} obs")

        return self.observations_df

    def extract_environmental_features(self, date: datetime = None):
        """
        Paso 3: Extraer features ambientales para todo el grid

        Parameters:
        -----------
        date : datetime, optional
            Fecha para features temporales (si None, usa fecha actual)
        """
        print_section_header("PASO 3: Extrayendo Features Ambientales", "🌱")

        date = date or datetime.now()

        self.features_df = self.feature_extractor.extract_features_for_grid(
            self.grid_df,
            date=date,
            add_interactions=False  # Añadiremos después de meteo
        )

        return self.features_df

    def add_meteorological_features(self):
        """
        Paso 4: Añadir features meteorológicas a observaciones

        CRÍTICO: Esto obtiene datos meteorológicos para cada observación
        en su fecha específica
        """
        print_section_header("PASO 4: Añadiendo Features Meteorológicas", "🌧️")

        # Para observaciones: obtener meteo de su fecha específica
        if self.observations_df is not None:
            logger.info("Obteniendo meteo para observaciones...")

            self.observations_df = self.meteo_fetcher.get_weather_for_observations(
                self.observations_df
            )

        return self.observations_df

    def perform_clustering(self):
        """
        Paso 5: Clustering ecológico
        """
        print_section_header("PASO 5: Clustering Ecológico", "🎲")

        self.features_df = self.clusterer.fit(self.features_df)

        # Añadir cluster a observaciones también
        if self.observations_df is not None:
            obs_with_cluster = self.observations_df.merge(
                self.features_df[['cell_id', 'cluster']],
                on='cell_id',
                how='left'
            )
            self.observations_df = obs_with_cluster

        return self.features_df

    def add_interaction_features(self):
        """
        Paso 6: Añadir features de interacción

        Se hace DESPUÉS de meteo para incluir interacciones ambientales-temporales
        """
        print_section_header("PASO 6: Features de Interacción", "🔗")

        self.features_df = self.feature_extractor.create_interaction_features(
            self.features_df
        )

        return self.features_df

    def train_models(self, species_list=None):
        """
        Paso 7: Entrenar modelos SDM para cada especie

        Parameters:
        -----------
        species_list : list, optional
            Lista de especies a entrenar. Si None, usa todas en config
        """
        print_section_header("PASO 7: Entrenando Modelos SDM", "🤖")

        if self.observations_df is None:
            logger.error("No hay observaciones cargadas")
            return None

        species_list = species_list or list(config.SPECIES_CONFIG.keys())

        for species in species_list:
            logger.info(f"\n{'='*60}")
            logger.info(f"Especie: {species}")
            logger.info(f"{'='*60}")

            # Crear modelo
            model = MushroomSDM(species)

            # Entrenar
            success = model.train(
                observations_df=self.observations_df,
                features_df=self.features_df,
                pseudo_absence_ratio=config.PSEUDO_ABSENCE_RATIO
            )

            if success:
                self.models[species] = model
                logger.info(f"✅ {species}: Entrenado")
            else:
                logger.warning(f"⚠️ {species}: Fallo en entrenamiento")

        logger.info(f"\n📊 Modelos entrenados: {len(self.models)}/{len(species_list)}")

        return self.models

    def predict_for_date(
        self,
        target_date: datetime,
        species: str = None,
        use_forecast: bool = False
    ) -> pd.DataFrame:
        """
        Predice probabilidades para una fecha específica

        Parameters:
        -----------
        target_date : datetime
            Fecha objetivo
        species : str, optional
            Especie a predecir (si None, predice todas)
        use_forecast : bool
            Si True, usa API forecast para fechas futuras

        Returns:
        --------
        pd.DataFrame : Predicciones con columnas [cell_id, lat, lon, probability, species]
        """
        print_section_header(
            f"Predicción para {target_date.strftime('%Y-%m-%d')}",
            "🔮"
        )

        # Obtener meteo para esta fecha
        logger.info("Obteniendo datos meteorológicos...")

        grid_with_meteo = self.meteo_fetcher.get_weather_for_grid(
            self.grid_df,
            target_date=target_date,
            use_forecast=use_forecast
        )

        if grid_with_meteo is None:
            logger.error("No se pudo obtener datos meteorológicos")
            return None

        # Extraer features (incluyendo temporales)
        logger.info("Extrayendo features...")

        features_for_date = self.feature_extractor.extract_features_for_grid(
            grid_with_meteo,
            date=target_date,
            meteo_df=grid_with_meteo,
            add_interactions=True
        )

        # Añadir clusters
        features_for_date = self.clusterer.predict(features_for_date)

        # Predecir
        species_list = [species] if species else list(self.models.keys())

        all_predictions = []

        for sp in species_list:
            if sp not in self.models:
                logger.warning(f"Modelo para {sp} no entrenado")
                continue

            logger.info(f"\nPrediciendo {sp}...")

            predictions = self.models[sp].predict(features_for_date)

            if predictions is not None:
                all_predictions.append(predictions)

        if len(all_predictions) == 0:
            return None

        # Combinar todas las predicciones
        result_df = pd.concat(all_predictions, ignore_index=True)

        logger.info(f"\n✅ Predicciones completadas: {len(result_df)} registros")

        return result_df

    def run_full_pipeline(
        self,
        observations_df: pd.DataFrame,
        species_list: list = None
    ) -> dict:
        """
        Ejecuta pipeline completo de entrenamiento

        Parameters:
        -----------
        observations_df : pd.DataFrame
            Observaciones de hongos
        species_list : list, optional
            Especies a entrenar

        Returns:
        --------
        dict : Resultados del pipeline
        """
        logger.info("\n" + "🍄"*30)
        logger.info("PIPELINE COMPLETO - INICIO")
        logger.info("🍄"*30 + "\n")

        # Pipeline steps
        self.setup_grid()
        self.load_observations(observations_df)
        self.extract_environmental_features()
        self.add_meteorological_features()
        self.perform_clustering()
        self.add_interaction_features()
        self.train_models(species_list)

        logger.info("\n" + "🍄"*30)
        logger.info("PIPELINE COMPLETO - FINALIZADO")
        logger.info("🍄"*30 + "\n")

        return {
            'grid': self.grid_df,
            'features': self.features_df,
            'observations': self.observations_df,
            'models': self.models
        }

    def save_pipeline(self, output_dir: str = 'outputs'):
        """
        Guarda todos los componentes del pipeline
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)

        # Grid
        self.grid_df.to_csv(output_path / 'grid.csv', index=False)

        # Features
        if self.features_df is not None:
            self.features_df.to_csv(output_path / 'features.csv', index=False)

        # Observations
        if self.observations_df is not None:
            self.observations_df.to_csv(output_path / 'observations.csv', index=False)

        # Modelos
        models_path = output_path / 'models'
        models_path.mkdir(exist_ok=True)

        for species, model in self.models.items():
            model_file = models_path / f"{species.replace(' ', '_')}.joblib"
            model.save_model(str(model_file))

        logger.info(f"✅ Pipeline guardado en: {output_path}")
