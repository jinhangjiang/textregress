"""
Base encoder interface for textregress.

This module defines the base class that all text encoders must implement.
"""

from abc import ABC, abstractmethod
from typing import List, Union, Optional
import torch

class BaseEncoder(ABC):
    """
    Abstract base class for all text encoders.
    
    This class defines the interface that all text encoders must implement.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the encoder.
        
        Args:
            **kwargs: Additional encoder-specific parameters.
        """
        self.fitted = False
    
    @abstractmethod
    def encode(self, text: str) -> Union[torch.Tensor, List[float]]:
        """
        Encode a single text string.
        
        Args:
            text (str): The text to encode.
            
        Returns:
            Union[torch.Tensor, List[float]]: The encoded text as either a tensor or a list of floats.
        """
        pass
    
    def fit(self, texts: List[str]) -> None:
        """
        Fit the encoder on a list of texts.
        
        This method is optional and should be implemented if the encoder needs to be fitted
        (e.g., for TF-IDF or other statistical encoders).
        
        Args:
            texts (List[str]): List of texts to fit on.
        """
        self.fitted = True
    
    @property
    @abstractmethod
    def output_dim(self) -> int:
        """
        Get the dimensionality of the encoder's output.
        
        Returns:
            int: The output dimensionality.
        """
        pass 