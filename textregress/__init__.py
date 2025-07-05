"""
TextRegress Package

A Python package for performing linear regression analysis on text data.
"""

__version__ = "1.2.3"

from .estimator import TextRegressor

# Import main components for easy access
from .models import (
    BaseTextRegressionModel,
    register_model,
    get_model,
    list_available_models
)

from .encoders import (
    BaseEncoder,
    register_encoder,
    get_encoder,
    list_available_encoders
)

from .losses import (
    BaseLoss,
    register_loss,
    get_loss_function,
    list_available_losses
)

from .utils import (
    chunk_text,
    pad_chunks,
    TextRegressionDataset,
    collate_fn
)

__all__ = [
    # Main estimator
    "TextRegressor",
    
    # Models
    "BaseTextRegressionModel",
    "register_model",
    "get_model", 
    "list_available_models",
    
    # Encoders
    "BaseEncoder",
    "register_encoder",
    "get_encoder",
    "list_available_encoders",
    
    # Losses
    "BaseLoss",
    "register_loss",
    "get_loss_function",
    "list_available_losses",
    
    # Utils
    "chunk_text",
    "pad_chunks",
    "TextRegressionDataset",
    "collate_fn"
]
