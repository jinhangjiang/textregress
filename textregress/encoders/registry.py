"""
Encoder registry system for textregress.

This module provides functionality to register and retrieve encoder implementations.
"""

from typing import Dict, Type, List
from .base import BaseEncoder

_ENCODER_REGISTRY: Dict[str, Type[BaseEncoder]] = {}

def register_encoder(name: str) -> callable:
    """
    Decorator to register an encoder class.
    
    Args:
        name (str): The name to register the encoder under.
        
    Returns:
        callable: The decorator function.
        
    Example:
        @register_encoder("sentence_transformer")
        class SentenceTransformerEncoder(BaseEncoder):
            pass
    """
    def decorator(encoder_cls: Type[BaseEncoder]) -> Type[BaseEncoder]:
        if name in _ENCODER_REGISTRY:
            raise ValueError(f"Encoder name '{name}' is already registered")
        if not issubclass(encoder_cls, BaseEncoder):
            raise TypeError(f"Encoder class must inherit from BaseEncoder")
        _ENCODER_REGISTRY[name] = encoder_cls
        return encoder_cls
    return decorator

def get_encoder(name: str, **kwargs) -> BaseEncoder:
    """
    Get a registered encoder instance by name.
    
    Args:
        name (str): The name of the registered encoder.
        **kwargs: Additional arguments to pass to the encoder constructor.
        
    Returns:
        BaseEncoder: An instance of the encoder.
        
    Raises:
        KeyError: If no encoder is registered under the given name.
    """
    if name not in _ENCODER_REGISTRY:
        raise KeyError(f"No encoder registered under name '{name}'. "
                      f"Available encoders: {list(_ENCODER_REGISTRY.keys())}")
    return _ENCODER_REGISTRY[name](**kwargs)

def list_available_encoders() -> List[str]:
    """
    List all registered encoder names.
    
    Returns:
        List[str]: List of registered encoder names.
    """
    return list(_ENCODER_REGISTRY.keys()) 