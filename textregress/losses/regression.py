"""
Regression loss function implementations.
"""

import torch
import torch.nn as nn
from typing import Optional
from .base import BaseLoss
from .registry import register_loss

@register_loss("mae")
class MAELoss(BaseLoss):
    """Mean Absolute Error loss."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.criterion = nn.L1Loss()
    
    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.criterion(pred, target)

@register_loss("mse")
class MSELoss(BaseLoss):
    """Mean Squared Error loss."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.criterion = nn.MSELoss()
    
    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.criterion(pred, target)

@register_loss("rmse")
class RMSELoss(BaseLoss):
    """Root Mean Squared Error loss."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.criterion = nn.MSELoss()
    
    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(self.criterion(pred, target))

@register_loss("smape")
class SMAPELoss(BaseLoss):
    """Symmetric Mean Absolute Percentage Error loss."""
    
    def __init__(self, epsilon: float = 1e-8, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon
    
    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        denominator = (torch.abs(pred) + torch.abs(target) + self.epsilon)
        return torch.mean(2.0 * torch.abs(pred - target) / denominator)

@register_loss("mape")
class MAPELoss(BaseLoss):
    """Mean Absolute Percentage Error loss."""
    
    def __init__(self, epsilon: float = 1e-8, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon
    
    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.abs((target - pred) / (target + self.epsilon)))

@register_loss("wmape")
class WMAPELoss(BaseLoss):
    """Weighted Mean Absolute Percentage Error loss."""
    
    def __init__(self, epsilon: float = 1e-8, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon
    
    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.sum(torch.abs(target - pred)) / (torch.sum(torch.abs(target)) + self.epsilon) 