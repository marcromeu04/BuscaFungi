"""
BuscaFungi - Ecological Clustering
Clustering ecológico para identificar nichos ambientales
"""

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import logging

from . import config

logger = logging.getLogger(__name__)


class EcologicalClusterer:
    """
    Clustering GMM para identificar zonas ecológicamente similares
    """

    def __init__(self, n_components=None):
        self.n_components = n_components or config.GMM_N_COMPONENTS
        self.gmm = None
        self.scaler = StandardScaler()
        self.feature_cols = None
        self.fitted = False

    def fit(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Ajusta GMM clustering a features ambientales

        Parameters:
        -----------
        features_df : pd.DataFrame
            Features ambientales del grid

        Returns:
        --------
        pd.DataFrame : features_df con columnas de cluster añadidas
        """
        logger.info(f"\n🎲 Clustering ecológico (GMM, n={self.n_components})")
        logger.info("="*60)

        # Seleccionar features para clustering
        # Usar solo features ambientales estables (no temporales)
        self.feature_cols = [
            'ph', 'elevation', 'twi', 'organic_carbon', 'slope',
            'clay_percent', 'sand_percent', 'aspect_sin', 'aspect_cos'
        ]

        # Añadir features de vegetación
        veg_cols = [col for col in features_df.columns if col.startswith('veg_')]
        self.feature_cols.extend(veg_cols)

        # Filtrar solo columnas que existen
        self.feature_cols = [
            col for col in self.feature_cols
            if col in features_df.columns
        ]

        logger.info(f"  Features para clustering: {len(self.feature_cols)}")

        X = features_df[self.feature_cols].copy()

        # Manejo de NaN
        if X.isnull().any().any():
            logger.warning("  NaN detectados. Imputando...")
            X = X.fillna(X.median())

        # Escalar
        X_scaled = self.scaler.fit_transform(X)

        # Fit GMM
        self.gmm = GaussianMixture(
            n_components=self.n_components,
            covariance_type=config.GMM_COVARIANCE_TYPE,
            random_state=42,
            max_iter=200,
            n_init=3
        )

        logger.info("  Entrenando GMM...")
        self.gmm.fit(X_scaled)

        # Predecir clusters
        clusters = self.gmm.predict(X_scaled)
        cluster_probs = self.gmm.predict_proba(X_scaled)

        # Añadir a features_df
        result_df = features_df.copy()
        result_df['cluster'] = clusters
        result_df['cluster_confidence'] = cluster_probs.max(axis=1)

        # Añadir probabilidades por cluster (útil para pseudo-ausencias)
        for i in range(self.n_components):
            result_df[f'cluster_{i}_prob'] = cluster_probs[:, i]

        # Estadísticas
        logger.info(f"\n  📊 Distribución de clusters:")
        cluster_counts = pd.Series(clusters).value_counts().sort_index()

        for cluster_id, count in cluster_counts.items():
            pct = count / len(clusters) * 100
            logger.info(f"     Cluster {cluster_id}: {count:6,} celdas ({pct:5.1f}%)")

        logger.info(f"\n  ✅ Clustering completado")

        self.fitted = True
        return result_df

    def predict(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Predice clusters para nuevas celdas
        """
        if not self.fitted:
            logger.error("Clusterer no entrenado")
            return None

        X = features_df[self.feature_cols].copy()

        if X.isnull().any().any():
            X = X.fillna(X.median())

        X_scaled = self.scaler.transform(X)

        clusters = self.gmm.predict(X_scaled)
        cluster_probs = self.gmm.predict_proba(X_scaled)

        result_df = features_df.copy()
        result_df['cluster'] = clusters
        result_df['cluster_confidence'] = cluster_probs.max(axis=1)

        for i in range(self.n_components):
            result_df[f'cluster_{i}_prob'] = cluster_probs[:, i]

        return result_df

    def get_cluster_characteristics(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Obtiene características promedio de cada cluster
        """
        if 'cluster' not in features_df.columns:
            logger.error("features_df no tiene columna 'cluster'")
            return None

        stats = []

        for cluster_id in sorted(features_df['cluster'].unique()):
            cluster_data = features_df[features_df['cluster'] == cluster_id]

            stats.append({
                'cluster': cluster_id,
                'n_cells': len(cluster_data),
                'elevation_mean': cluster_data['elevation'].mean(),
                'elevation_std': cluster_data['elevation'].std(),
                'ph_mean': cluster_data['ph'].mean(),
                'organic_carbon_mean': cluster_data['organic_carbon'].mean(),
                'twi_mean': cluster_data['twi'].mean(),
                'slope_mean': cluster_data['slope'].mean(),
            })

        stats_df = pd.DataFrame(stats)
        return stats_df
