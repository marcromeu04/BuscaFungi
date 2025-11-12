"""
BuscaFungi - Sistema de Predicción de Hongos
Sistema profesional de predicción espacio-temporal de hongos comestibles en España
"""

__version__ = "2.0.0"
__author__ = "BuscaFungi Team"

from . import config
from .grid import GridManager
from .features import FeatureExtractor
from .meteo import MeteoDataFetcher
from .clustering import EcologicalClusterer
from .sdm import MushroomSDM
from .pipeline import BuscaFungiPipeline
from . import utils
from .pseudo_absences import generate_smart_pseudo_absences

__all__ = [
    'config',
    'GridManager',
    'FeatureExtractor',
    'MeteoDataFetcher',
    'EcologicalClusterer',
    'MushroomSDM',
    'BuscaFungiPipeline',
    'utils',
    'generate_smart_pseudo_absences'
]
