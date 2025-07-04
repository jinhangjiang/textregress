"""
Models package for textregress.

This package contains all model implementations for text regression tasks.
Models can be registered using the @register_model decorator and retrieved
using the get_model function.
"""

from .base import BaseTextRegressionModel
from .registry import register_model, get_model, list_available_models

# Import all model implementations to ensure they're registered
from .lstm import LSTMTextRegressionModel
from .gru import GRUTextRegressionModel

__all__ = [
    'BaseTextRegressionModel',
    'LSTMTextRegressionModel',
    'GRUTextRegressionModel',
    'register_model',
    'get_model',
    'list_available_models',
] 