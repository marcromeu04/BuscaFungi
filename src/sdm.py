"""
BuscaFungi - Species Distribution Model
Modelo de distribución de especies con validación espacial
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import logging
from typing import Optional, List

from . import config
from .utils import create_spatial_groups
from .pseudo_absences import generate_smart_pseudo_absences

logger = logging.getLogger(__name__)


class MushroomSDM:
    """
    Species Distribution Model para hongos

    Mejoras sobre versión original:
    - Sin data leakage (usa grid fijo)
    - Features temporales integradas
    - Validación espacial robusta
    - Pseudo-ausencias inteligentes
    """

    def __init__(self, species_name: str):
        """
        Parameters:
        -----------
        species_name : str
            Nombre científico de la especie (debe estar en config.SPECIES_CONFIG)
        """
        self.species = species_name

        if species_name not in config.SPECIES_CONFIG:
            raise ValueError(f"Especie desconocida: {species_name}")

        self.config = config.SPECIES_CONFIG[species_name]

        self.model = None
        self.scaler = StandardScaler()
        self.feature_cols = None
        self.feature_importance = None
        self.trained = False

        logger.info(f"MushroomSDM inicializado: {species_name}")

    def _select_features(self, df: pd.DataFrame) -> List[str]:
        """
        Selecciona features para el modelo

        Excluye:
        - IDs y metadatos
        - Coordenadas (para evitar overfitting espacial)
        - Categorías no codificadas
        """
        exclude_patterns = [
            'cell_id', 'id', 'lat', 'lon', 'grid_lat', 'grid_lon',
            'date', 'species', 'observed', 'vegetation_type',
            'snap_distance', 'cluster'  # Cluster se usa en pseudo-ausencias, no en modelo
        ]

        feature_cols = [
            col for col in df.columns
            if not any(pattern in col.lower() for pattern in exclude_patterns)
            and df[col].dtype in ['float64', 'int64', 'float32', 'int32']
        ]

        logger.info(f"  Features seleccionadas: {len(feature_cols)}")
        logger.debug(f"  {feature_cols}")

        return feature_cols

    def train(
        self,
        observations_df: pd.DataFrame,
        features_df: pd.DataFrame,
        pseudo_absence_ratio: float = 2.0
    ) -> bool:
        """
        Entrena el modelo SDM

        Parameters:
        -----------
        observations_df : pd.DataFrame
            Observaciones de la especie (con 'cell_id' asignado al grid)
        features_df : pd.DataFrame
            Features del grid completo (debe incluir todas las celdas)
        pseudo_absence_ratio : float
            Ratio ausencias:presencias

        Returns:
        --------
        bool : True si entrenamiento exitoso
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"🤖 Entrenando modelo: {self.species}")
        logger.info(f"{'='*60}")

        # Filtrar observaciones de esta especie
        presences = observations_df[
            observations_df['species'] == self.species
        ].copy()

        if len(presences) < 20:
            logger.warning(f"  ⚠️ Solo {len(presences)} observaciones. Mínimo: 20")
            return False

        logger.info(f"  ✅ {len(presences)} observaciones")

        # Obtener features de las presencias usando cell_id
        if 'cell_id' not in presences.columns:
            logger.error("  ❌ observations_df debe tener 'cell_id'")
            return False

        if 'cell_id' not in features_df.columns:
            logger.error("  ❌ features_df debe tener 'cell_id'")
            return False

        # Merge presencias con features
        presences_with_features = presences.merge(
            features_df,
            on='cell_id',
            how='inner',
            suffixes=('_obs', '_grid')
        )

        if len(presences_with_features) == 0:
            logger.error("  ❌ No se pudo hacer match presencias-features")
            return False

        n_matched = len(presences_with_features)
        logger.info(f"  ✅ {n_matched}/{len(presences)} presencias con features")

        if n_matched < 20:
            logger.warning(f"  ⚠️ Muy pocas presencias con features (<20)")
            return False

        # Generar pseudo-ausencias
        logger.info(f"\n  🎲 Generando pseudo-ausencias...")

        absence_indices = generate_smart_pseudo_absences(
            presences_df=presences_with_features,
            grid_df=features_df,
            ratio=pseudo_absence_ratio,
            min_distance_km=config.MIN_DISTANCE_KM,
            cluster_col='cluster',
            use_cluster_filter=True
        )

        if len(absence_indices) < n_matched:
            logger.warning(f"  ⚠️ Pocas ausencias generadas ({len(absence_indices)})")
            return False

        # Preparar dataset
        X_presence = features_df.loc[
            features_df['cell_id'].isin(presences_with_features['cell_id'])
        ].copy()
        X_absence = features_df.loc[absence_indices].copy()

        # Seleccionar features
        self.feature_cols = self._select_features(features_df)

        X_pres = X_presence[self.feature_cols]
        X_abs = X_absence[self.feature_cols]

        # Combinar
        X = pd.concat([X_pres, X_abs], ignore_index=True)
        y = np.array([1] * len(X_pres) + [0] * len(X_abs))

        logger.info(f"\n  📊 Dataset final:")
        logger.info(f"     Presencias: {len(X_pres)}")
        logger.info(f"     Ausencias:  {len(X_abs)}")
        logger.info(f"     Total:      {len(X)}")
        logger.info(f"     Features:   {len(self.feature_cols)}")

        # Manejo de NaN
        if X.isnull().any().any():
            logger.warning("  ⚠️ NaN detectados. Imputando con mediana...")
            X = X.fillna(X.median())

        # Escalar
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            columns=X.columns
        )

        # Coordenadas para validación espacial
        lat_coords = pd.concat([
            X_presence['lat'],
            X_absence['lat']
        ], ignore_index=True)

        lon_coords = pd.concat([
            X_presence['lon'],
            X_absence['lon']
        ], ignore_index=True)

        # Validación cruzada espacial
        logger.info(f"\n  🔄 Validación espacial ({config.N_SPATIAL_FOLDS} folds)...")

        spatial_df = pd.DataFrame({'lat': lat_coords, 'lon': lon_coords})
        spatial_groups = create_spatial_groups(
            spatial_df,
            block_size=config.SPATIAL_BLOCK_SIZE_DEG
        )

        auc_scores = self._spatial_cross_validation(X_scaled, y, spatial_groups)

        logger.info(f"  📈 AUC-ROC: {auc_scores.mean():.3f} ± {auc_scores.std():.3f}")
        logger.info(f"     Range: [{auc_scores.min():.3f}, {auc_scores.max():.3f}]")

        # Entrenar modelo final
        logger.info(f"\n  🏗️ Entrenando modelo final...")

        self.model = xgb.XGBClassifier(**config.XGBOOST_PARAMS)
        self.model.fit(X_scaled, y)

        # Feature importance
        self.feature_importance = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        logger.info(f"\n  🔝 Top 10 features:")
        for idx, row in self.feature_importance.head(10).iterrows():
            logger.info(f"     {row['feature']:30} {row['importance']:.3f}")

        self.trained = True
        logger.info(f"\n  ✅ Modelo entrenado exitosamente")

        return True

    def _spatial_cross_validation(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        groups: pd.Series
    ) -> np.ndarray:
        """
        Validación cruzada espacial (GroupKFold)
        """
        gkf = GroupKFold(n_splits=config.N_SPATIAL_FOLDS)
        scores = []

        for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Entrenar fold
            fold_model = xgb.XGBClassifier(**config.XGBOOST_PARAMS)
            fold_model.fit(X_train, y_train)

            # Predecir
            y_pred_proba = fold_model.predict_proba(X_test)[:, 1]

            # AUC-ROC
            score = roc_auc_score(y_test, y_pred_proba)
            scores.append(score)

            logger.debug(f"    Fold {fold+1}: AUC = {score:.3f}")

        return np.array(scores)

    def predict(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Predice probabilidades para un grid

        Parameters:
        -----------
        features_df : pd.DataFrame
            Features del grid

        Returns:
        --------
        pd.DataFrame : Grid con columna 'probability' (0-100)
        """
        if not self.trained:
            logger.error("Modelo no entrenado")
            return None

        logger.info(f"Prediciendo para {len(features_df)} celdas...")

        # Seleccionar features
        X = features_df[self.feature_cols].copy()

        # Manejo de NaN
        if X.isnull().any().any():
            logger.warning("  NaN detectados en predicción. Imputando...")
            X = X.fillna(X.median())

        # Escalar
        X_scaled = pd.DataFrame(
            self.scaler.transform(X),
            columns=X.columns,
            index=X.index
        )

        # Predecir
        probabilities = self.model.predict_proba(X_scaled)[:, 1]

        # Convertir a 0-100
        probabilities = probabilities * 100

        # Crear resultado
        result_df = features_df[['cell_id', 'lat', 'lon']].copy()
        result_df['probability'] = probabilities
        result_df['species'] = self.species

        logger.info(f"✅ Predicción completada")
        logger.info(f"   Media: {probabilities.mean():.1f}%")
        logger.info(f"   Celdas >50%: {(probabilities > 50).sum()}")
        logger.info(f"   Celdas >75%: {(probabilities > 75).sum()}")

        return result_df

    def save_model(self, filepath: str):
        """Guarda modelo entrenado"""
        import joblib
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_cols': self.feature_cols,
            'species': self.species
        }, filepath)
        logger.info(f"Modelo guardado: {filepath}")

    def load_model(self, filepath: str):
        """Carga modelo entrenado"""
        import joblib
        data = joblib.load(filepath)
        self.model = data['model']
        self.scaler = data['scaler']
        self.feature_cols = data['feature_cols']
        self.species = data['species']
        self.trained = True
        logger.info(f"Modelo cargado: {filepath}")
