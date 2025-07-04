"""
Loss functions package for textregress.

This package contains all loss function implementations.
Loss functions can be registered using the @register_loss decorator and retrieved
using the get_loss_function function.
"""

from .base import BaseLoss
from .registry import register_loss, get_loss_function, list_available_losses

# Import all loss implementations to ensure they're registered
from .regression import (
    MAELoss,
    MSELoss,
    RMSELoss,
    SMAPELoss,
    MAPELoss,
    WMAPELoss
)

__all__ = [
    'BaseLoss',
    'MAELoss',
    'MSELoss',
    'RMSELoss',
    'SMAPELoss',
    'MAPELoss',
    'WMAPELoss',
    'register_loss',
    'get_loss_function',
    'list_available_losses',
] 