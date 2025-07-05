"""
Dataset utilities for text regression.

This module contains classes and functions for handling text regression datasets.
"""

from typing import Dict, List, Optional, Tuple, Union
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


class TextRegressionDataset(torch.utils.data.Dataset):
    """
    Dataset for text regression.
    
    Each sample is a sequence of encoded chunks and a target value.
    Optionally includes exogenous features.
    """
    def __init__(self, encoded_sequences, targets, exogenous=None):
        """
        Args:
            encoded_sequences (list of list of torch.Tensor): Each inner list contains encoded chunks for a text sample.
            targets (list or array-like): Target values.
            exogenous (list, optional): List of exogenous feature vectors.
        """
        self.encoded_sequences = encoded_sequences
        self.targets = list(targets)
        self.exogenous = exogenous

    def __len__(self):
        return len(self.encoded_sequences)

    def __getitem__(self, idx):
        if self.exogenous is not None:
            return self.encoded_sequences[idx], self.exogenous[idx], self.targets[idx]
        else:
            return self.encoded_sequences[idx], self.targets[idx]

def collate_fn(batch):
    """
    Custom collate function to pad sequences of encoded chunks.
    
    Args:
        batch (list): List of samples, each sample is either (encoded_sequence, target) or (encoded_sequence, exogenous, target).
    
    Returns:
        Tuple: Padded sequences, (exogenous features if available), and targets.
    """
    # Check if exogenous features are provided.
    if len(batch[0]) == 3:
        sequences, exogenous, targets = zip(*batch)
    else:
        sequences, targets = zip(*batch)
        exogenous = None
    
    # Each item in sequences is a list of torch.Tensor; pad them along the sequence dimension.
    sequence_tensors = [torch.stack(seq) for seq in sequences]  # shape: (num_chunks, feature_dim)
    padded_sequences = pad_sequence(sequence_tensors, batch_first=True)  # shape: (batch_size, max_chunks, feature_dim)
    
    targets = torch.tensor(targets, dtype=torch.float32)
    
    if exogenous is not None:
        exogenous = torch.tensor(exogenous, dtype=torch.float32)
        return padded_sequences, exogenous, targets
    else:
        return padded_sequences, targets 