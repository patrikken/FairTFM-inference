"""
FairTFM - Fair Transformer Foundation Model for Tabular Data
Inference-only repository for making fair predictions on tabular data.
"""

from .model import FairTFM
from .inference import FairTFMClassifier
from .utils import load_model_from_checkpoint, get_default_device, set_randomness_seed, compute_fairness_metrics
from .datasets.loaders import ACSPumsDataset

__version__ = "1.0.0"
__all__ = [
    "FairTFM",
    "FairTFMClassifier",
    "load_model_from_checkpoint",
    "get_default_device",
    "set_randomness_seed",
    "ACSPumsDataset", 
    "compute_fairness_metrics"
]
