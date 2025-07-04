"""
Dataset utilities for text regression.

This module contains classes and functions for handling text regression datasets.
"""

from typing import Dict, List, Optional, Tuple, Union
import torch
from torch.utils.data import Dataset, DataLoader


class TextRegressionDataset(Dataset):
    """
    Dataset for text regression tasks.
    
    This dataset handles text data and corresponding regression targets,
    with optional support for additional features.
    """
    
    def __init__(
        self,
        texts: List[str],
        targets: List[float],
        encoder,
        max_length: Optional[int] = None,
        additional_features: Optional[Dict[str, List[float]]] = None
    ):
        """
        Initialize the dataset.
        
        Args:
            texts: List of input texts
            targets: List of target values
            encoder: Text encoder to use
            max_length: Maximum sequence length (optional)
            additional_features: Dictionary of additional feature names and values
        """
        self.texts = texts
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.encoder = encoder
        self.max_length = max_length
        self.additional_features = additional_features
        
        if additional_features is not None:
            self.feature_names = list(additional_features.keys())
            self.feature_values = torch.tensor(
                [additional_features[name] for name in self.feature_names],
                dtype=torch.float32
            ).T
        else:
            self.feature_names = []
            self.feature_values = None
    
    def __len__(self) -> int:
        """Get the number of samples in the dataset."""
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Index of the sample to get
            
        Returns:
            Dictionary containing encoded text, target, and optional features
        """
        text = self.texts[idx]
        target = self.targets[idx]
        
        # Encode text
        encoded = self.encoder.encode(text, max_length=self.max_length)
        
        # Prepare output
        output = {
            'text': encoded,
            'target': target
        }
        
        # Add additional features if present
        if self.additional_features is not None:
            output['features'] = self.feature_values[idx]
            
        return output


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Collate function for DataLoader.
    
    Args:
        batch: List of samples from the dataset
        
    Returns:
        Dictionary of batched tensors
    """
    # Get all keys from the first sample
    keys = batch[0].keys()
    
    # Initialize output dictionary
    output = {}
    
    # Process each key
    for key in keys:
        if key == 'text':
            # Stack text encodings
            output[key] = torch.stack([item[key] for item in batch])
        elif key == 'features':
            # Stack additional features
            output[key] = torch.stack([item[key] for item in batch])
        else:
            # Stack other tensors (e.g., targets)
            output[key] = torch.stack([item[key] for item in batch])
            
    return output 