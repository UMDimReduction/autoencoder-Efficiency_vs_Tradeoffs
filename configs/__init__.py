"""
================================================================================
Configuration Package
================================================================================
Paper: "A comprehensive study of auto-encoders for anomaly detection"
Authors: Asif Ahmed Neloy, Maxime Turgeon

Contains all hyperparameters and expected results from the paper.

Usage:
    from configs import DEVICE, MODEL_CONFIGS, EXPECTED_RESULTS
================================================================================
"""

from .config import (
    DEVICE,
    DATASET_CONFIG,
    TRAINING_CONFIG,
    MODEL_CONFIGS,
    EXPECTED_RESULTS,
    TRAINING_TIMES
)

__all__ = [
    'DEVICE',
    'DATASET_CONFIG',
    'TRAINING_CONFIG',
    'MODEL_CONFIGS',
    'EXPECTED_RESULTS',
    'TRAINING_TIMES',
]
