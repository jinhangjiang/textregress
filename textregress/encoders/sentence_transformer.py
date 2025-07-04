"""
Sentence Transformer encoder implementation for textregress.

This module provides a sentence transformer-based encoder using the sentence-transformers library.
"""

from typing import List, Union
import torch
from sentence_transformers import SentenceTransformer
from .base import BaseEncoder
from .registry import register_encoder


@register_encoder("sentence-transformers/all-MiniLM-L6-v2")
class SentenceTransformerEncoder(BaseEncoder):
    """
    Sentence Transformer encoder using the sentence-transformers library.
    
    This encoder uses pre-trained sentence transformer models to encode text into
    high-dimensional vectors suitable for downstream tasks.
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", **kwargs):
        """
        Initialize the Sentence Transformer encoder.
        
        Args:
            model_name (str): The name of the pre-trained model to use.
            **kwargs: Additional arguments passed to the SentenceTransformer constructor.
        """
        super().__init__(**kwargs)
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, **kwargs)
        self.fitted = True  # Sentence transformers are pre-fitted
    
    def encode(self, text: str) -> Union[torch.Tensor, List[float]]:
        """
        Encode a single text string using the sentence transformer.
        
        Args:
            text (str): The text to encode.
            
        Returns:
            Union[torch.Tensor, List[float]]: The encoded text as a tensor or list of floats.
        """
        if not text or not text.strip():
            # Return zero vector for empty text
            return torch.zeros(self.output_dim)
        
        # Encode the text
        embedding = self.model.encode(text, convert_to_tensor=True)
        
        # Convert to tensor if it's not already
        if isinstance(embedding, torch.Tensor):
            return embedding
        else:
            return torch.tensor(embedding, dtype=torch.float32)
    
    def fit(self, texts: List[str]) -> None:
        """
        Fit the encoder on a list of texts.
        
        Note: Sentence transformers are pre-trained, so this method doesn't actually
        fit anything, but it marks the encoder as fitted.
        
        Args:
            texts (List[str]): List of texts to fit on (ignored for sentence transformers).
        """
        # Sentence transformers are pre-trained, so no actual fitting is needed
        self.fitted = True
    
    @property
    def output_dim(self) -> int:
        """
        Get the dimensionality of the encoder's output.
        
        Returns:
            int: The output dimensionality.
        """
        return self.model.get_sentence_embedding_dimension() 