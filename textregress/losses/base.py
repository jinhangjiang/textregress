"""
Base loss function interface for textregress.

This module defines the base class that all loss functions must implement.
"""

from abc import ABC, abstractmethod
import torch
from typing import Union, Tuple

class BaseLoss(ABC):
    """
    Abstract base class for all loss functions.
    
    This class defines the interface that all loss functions must implement.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the loss function.
        
        Args:
            **kwargs: Additional loss-specific parameters.
        """
        pass
    
    @abstractmethod
    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the loss between predictions and targets.
        
        Args:
            pred (torch.Tensor): Model predictions.
            target (torch.Tensor): Target values.
            
        Returns:
            torch.Tensor: The computed loss value.
        """
        pass 