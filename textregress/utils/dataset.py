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
    
    This dataset handles pre-encoded text sequences and corresponding regression targets,
    with optional support for exogenous features.
    """
    
    def __init__(
        self,
        encoded_sequences: List[List[torch.Tensor]],
        targets: List[float],
        exogenous: Optional[List[List[float]]] = None
    ):
        """
        Initialize the dataset.
        
        Args:
            encoded_sequences: List of lists of encoded text tensors (each inner list represents chunks)
            targets: List of target values
            exogenous: List of exogenous feature lists (optional)
        """
        self.encoded_sequences = encoded_sequences
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.exogenous = exogenous
        
        if exogenous is not None:
            self.exogenous_tensor = torch.tensor(exogenous, dtype=torch.float32)
        else:
            self.exogenous_tensor = None
    
    def __len__(self) -> int:
        """Get the number of samples in the dataset."""
        return len(self.encoded_sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Index of the sample to get
            
        Returns:
            Dictionary containing encoded sequences, target, and optional exogenous features
        """
        # Get the encoded sequences for this sample
        sequences = self.encoded_sequences[idx]
        
        # Stack the sequences into a single tensor
        # Each sequence is a tensor of shape (seq_len, features)
        # We stack them to get (num_chunks, seq_len, features)
        x = torch.stack(sequences)
        
        # Get target
        y = self.targets[idx]
        
        # Prepare output
        output = {
            'x': x,
            'y': y
        }
        
        # Add exogenous features if present
        if self.exogenous_tensor is not None:
            output['exogenous'] = self.exogenous_tensor[idx]
            
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
        if key == 'x':
            # For sequences, we need to handle variable lengths
            # Pad sequences to the same length within the batch
            max_chunks = max(item[key].shape[0] for item in batch)
            max_seq_len = max(item[key].shape[1] for item in batch)
            
            # Check if we have a feature dimension
            if len(batch[0][key].shape) > 2:
                feature_dim = batch[0][key].shape[2]
                padded_sequences = torch.zeros(len(batch), max_chunks, max_seq_len, feature_dim)
            else:
                # Handle 2D tensors (no feature dimension)
                padded_sequences = torch.zeros(len(batch), max_chunks, max_seq_len)
            
            for i, item in enumerate(batch):
                seq = item[key]
                if len(seq.shape) > 2:
                    padded_sequences[i, :seq.shape[0], :seq.shape[1], :] = seq
                else:
                    padded_sequences[i, :seq.shape[0], :seq.shape[1]] = seq
            
            output[key] = padded_sequences
        else:
            # Stack other tensors (e.g., targets, exogenous features)
            output[key] = torch.stack([item[key] for item in batch])
            
    return output 