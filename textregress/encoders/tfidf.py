"""
TF-IDF encoder implementation for textregress.

This module provides a TF-IDF based encoder using scikit-learn.
The TF-IDF encoder learns the vocabulary from all chunks across all documents
and then transforms individual chunks using the learned vocabulary.
"""

from typing import List, Union
import torch
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from .base import BaseEncoder
from .registry import register_encoder


@register_encoder("tfidf")
class TfidfEncoder(BaseEncoder):
    """
    TF-IDF encoder using scikit-learn's TfidfVectorizer.
    
    This encoder implements TF-IDF of chunks (TF-IDF-C) where:
    1. All chunks from all documents are used to learn the vocabulary and IDF values
    2. Individual chunks are then transformed using the learned vocabulary
    3. This ensures proper TF-IDF statistics across the entire corpus
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the TF-IDF encoder.
        
        Args:
            **kwargs: Additional arguments passed to TfidfVectorizer.
        """
        super().__init__(**kwargs)
        self.vectorizer = TfidfVectorizer(**kwargs)
        self.fitted = False
        self._all_chunks = []  # Store all chunks for fitting
    
    def encode(self, text: str) -> Union[torch.Tensor, List[float]]:
        """
        Encode a single text chunk using TF-IDF.
        
        Args:
            text (str): The text chunk to encode.
            
        Returns:
            Union[torch.Tensor, List[float]]: The encoded text as a tensor or list of floats.
        """
        if not self.fitted:
            raise ValueError("Encoder must be fitted before encoding. Call fit() with all chunks first.")
        
        if not text or not text.strip():
            # Return zero vector for empty text
            return torch.zeros(self.output_dim)
        
        # Transform the chunk using the fitted vectorizer
        features = self.vectorizer.transform([text]).toarray()[0]
        
        # Convert to tensor
        return torch.tensor(features, dtype=torch.float32)
    
    def fit(self, texts: List[str]) -> None:
        """
        Fit the encoder on all chunks from all documents.
        
        This method should be called with ALL chunks from ALL documents to learn
        the proper vocabulary and IDF values for TF-IDF calculation.
        
        Args:
            texts (List[str]): List of all text chunks from all documents.
        """
        # Store all chunks for reference
        self._all_chunks = texts
        
        # Fit the vectorizer on all chunks to learn vocabulary and IDF
        self.vectorizer.fit(texts)
        self.fitted = True
        
        print(f"TF-IDF encoder fitted on {len(texts)} chunks with vocabulary size: {self.output_dim}")
    
    @property
    def output_dim(self) -> int:
        """
        Get the dimensionality of the encoder's output.
        
        Returns:
            int: The output dimensionality (vocabulary size).
        """
        if not self.fitted:
            return 1000  # Default dimension before fitting
        return len(self.vectorizer.get_feature_names_out())
    
    def get_feature_names(self) -> List[str]:
        """
        Get the feature names (vocabulary) learned during fitting.
        
        Returns:
            List[str]: List of feature names.
        """
        if not self.fitted:
            return []
        return self.vectorizer.get_feature_names_out().tolist() 