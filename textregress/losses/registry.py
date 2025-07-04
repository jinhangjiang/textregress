"""
Loss function registry system for textregress.

This module provides functionality to register and retrieve loss function implementations.
"""

from typing import Dict, Type, List, Callable, Union
from .base import BaseLoss

_LOSS_REGISTRY: Dict[str, Union[Type[BaseLoss], Callable]] = {}

def register_loss(name: str) -> callable:
    """
    Decorator to register a loss function.
    
    Args:
        name (str): The name to register the loss function under.
        
    Returns:
        callable: The decorator function.
        
    Example:
        @register_loss("mae")
        class MAELoss(BaseLoss):
            pass
    """
    def decorator(loss_cls: Union[Type[BaseLoss], Callable]) -> Union[Type[BaseLoss], Callable]:
        if name in _LOSS_REGISTRY:
            raise ValueError(f"Loss function name '{name}' is already registered")
        if isinstance(loss_cls, type) and not issubclass(loss_cls, BaseLoss):
            raise TypeError(f"Loss class must inherit from BaseLoss")
        _LOSS_REGISTRY[name] = loss_cls
        return loss_cls
    return decorator

def get_loss_function(name: str) -> Union[Type[BaseLoss], Callable]:
    """
    Get a registered loss function by name.
    
    Args:
        name (str): The name of the registered loss function.
        
    Returns:
        Union[Type[BaseLoss], Callable]: The loss function class or callable.
        
    Raises:
        KeyError: If no loss function is registered under the given name.
    """
    if name not in _LOSS_REGISTRY:
        raise KeyError(f"No loss function registered under name '{name}'. "
                      f"Available loss functions: {list(_LOSS_REGISTRY.keys())}")
    return _LOSS_REGISTRY[name]

def list_available_losses() -> List[str]:
    """
    List all registered loss function names.
    
    Returns:
        List[str]: List of registered loss function names.
    """
    return list(_LOSS_REGISTRY.keys()) 