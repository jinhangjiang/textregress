"""
Utility functions package for textregress.

This package contains various utility functions and classes used throughout the library.
"""

from .text import chunk_text, pad_chunks
from .dataset import TextRegressionDataset, collate_fn
from .explainability import (
    get_attention_weights,
    get_gradient_importance,
    integrated_gradients,
)

__all__ = [
    'chunk_text',
    'pad_chunks',
    'TextRegressionDataset',
    'collate_fn',
    'get_attention_weights',
    'get_gradient_importance',
    'integrated_gradients',
] 
