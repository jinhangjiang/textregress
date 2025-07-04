"""
Mock encoder implementation for testing textregress.

This module provides a simple mock encoder for testing purposes that doesn't require
external dependencies like sentence-transformers.
"""

from typing import List, Union
import torch
import numpy as np
from .base import BaseEncoder
from .registry import register_encoder


@register_encoder("mock-encoder")
class MockEncoder(BaseEncoder):
    """
    Mock encoder for testing purposes.
    
    This encoder provides a simple implementation that doesn't require external
    dependencies. It generates random embeddings for testing.
    """
    
    def __init__(self, output_dim: int = 384, **kwargs):
        """
        Initialize the mock encoder.
        
        Args:
            output_dim (int): The dimensionality of the output embeddings.
            **kwargs: Additional arguments (ignored).
        """
        super().__init__(**kwargs)
        self._output_dim = output_dim
        self.fitted = True
    
    def encode(self, text: str) -> Union[torch.Tensor, List[float]]:
        """
        Encode a single text string using random embeddings.
        
        Args:
            text (str): The text to encode (ignored for mock encoder).
            
        Returns:
            Union[torch.Tensor, List[float]]: A random embedding.
        """
        # Generate a random embedding based on the text length for reproducibility
        if not text:
            embedding = np.zeros(self._output_dim)
        else:
            # Use text length as seed for reproducible random embeddings
            np.random.seed(len(text))
            embedding = np.random.randn(self._output_dim)
            embedding = embedding / np.linalg.norm(embedding)  # Normalize
        
        return torch.tensor(embedding, dtype=torch.float32)
    
    def fit(self, texts: List[str]) -> None:
        """
        Fit the encoder on a list of texts.
        
        Args:
            texts (List[str]): List of texts to fit on (ignored for mock encoder).
        """
        self.fitted = True
    
    @property
    def output_dim(self) -> int:
        """
        Get the dimensionality of the encoder's output.
        
        Returns:
            int: The output dimensionality.
        """
        return self._output_dim 