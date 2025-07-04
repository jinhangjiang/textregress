"""
Model registry system for textregress.

This module provides functionality to register and retrieve model implementations.
"""

from typing import Dict, Type, List, Optional
from .base import BaseTextRegressionModel

_MODEL_REGISTRY: Dict[str, Type[BaseTextRegressionModel]] = {}

def register_model(name: str) -> callable:
    """
    Decorator to register a model class.
    
    Args:
        name (str): The name to register the model under.
        
    Returns:
        callable: The decorator function.
        
    Example:
        @register_model("lstm")
        class LSTMTextRegressionModel(BaseTextRegressionModel):
            pass
    """
    def decorator(model_cls: Type[BaseTextRegressionModel]) -> Type[BaseTextRegressionModel]:
        if name in _MODEL_REGISTRY:
            raise ValueError(f"Model name '{name}' is already registered")
        if not issubclass(model_cls, BaseTextRegressionModel):
            raise TypeError(f"Model class must inherit from BaseTextRegressionModel")
        _MODEL_REGISTRY[name] = model_cls
        return model_cls
    return decorator

def get_model(name: str) -> Type[BaseTextRegressionModel]:
    """
    Get a registered model class by name.
    
    Args:
        name (str): The name of the registered model.
        
    Returns:
        Type[BaseTextRegressionModel]: The model class.
        
    Raises:
        KeyError: If no model is registered under the given name.
    """
    if name not in _MODEL_REGISTRY:
        raise KeyError(f"No model registered under name '{name}'. "
                      f"Available models: {list(_MODEL_REGISTRY.keys())}")
    return _MODEL_REGISTRY[name]

def list_available_models() -> List[str]:
    """
    List all registered model names.
    
    Returns:
        List[str]: List of registered model names.
    """
    return list(_MODEL_REGISTRY.keys()) 