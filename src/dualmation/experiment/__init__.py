"""
Experimentation and research infrastructure for DualAnimate.

Provides:
- Multi-backend experiment tracking (TensorBoard, W&B, CSV/JSON)
- Structured logging with file + console handlers
- OmegaConf-based config management for reproducible experiments
- Seed management & config snapshot utilities
- Metrics collection and paper-ready visualization
"""

from dualmation.experiment.tracker import ExperimentTracker
from dualmation.experiment.config import ExperimentConfig, load_config, save_config
from dualmation.experiment.reproducibility import set_seed, get_system_info
from dualmation.experiment.metrics import MetricsCollector

__all__ = [
    "ExperimentTracker",
    "ExperimentConfig",
    "load_config",
    "save_config",
    "set_seed",
    "get_system_info",
    "MetricsCollector",
]
