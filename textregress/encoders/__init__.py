"""
Encoders package for textregress.

This package contains all text encoder implementations.
Encoders can be registered using the @register_encoder decorator and retrieved
using the get_encoder function.
"""

from .base import BaseEncoder
from .registry import register_encoder, get_encoder, list_available_encoders

# Import all encoder implementations to ensure they're registered
try:
    from .sentence_transformer import SentenceTransformerEncoder
except ImportError:
    # Fallback to mock encoder if sentence-transformers is not available
    from .mock import MockEncoder as SentenceTransformerEncoder

__all__ = [
    'BaseEncoder',
    'SentenceTransformerEncoder',
    'MockEncoder',
    'register_encoder',
    'get_encoder',
    'list_available_encoders',
] 