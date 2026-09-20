"""Reproducible research utilities for power-grid intrusion detection."""

from .features import FEATURE_GROUPS, build_features
from .simulation import ATTACK_CLASSES, SimulationConfig, generate_dataset

__all__ = [
    "ATTACK_CLASSES",
    "FEATURE_GROUPS",
    "SimulationConfig",
    "build_features",
    "generate_dataset",
]

__version__ = "2.0.0"
