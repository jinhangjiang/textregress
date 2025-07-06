"""
Lightweight explainability utilities for textregress models.

Provides:
- Gradient-based feature importance (saliency)
- Attention weights extraction (for cross-attention models)
"""

import torch
from typing import Optional, Dict, Any

def get_gradient_importance(model, x: torch.Tensor, exogenous: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
    """
    Compute gradient-based feature importance for input text and exogenous features.
    Args:
        model: The textregress model
        x: Input tensor (batch_size, seq_len, features)
        exogenous: Exogenous features (batch_size, n_features) or None
    Returns:
        Dict with 'text_importance' and optionally 'exogenous_importance'
    """
    # Store original training state
    was_training = model.training
    
    # Set to training mode for gradient computation (required for cuDNN RNN)
    model.train()
    
    # Ensure tensors are on the same device as the model
    device = next(model.parameters()).device
    x = x.to(device).clone().detach().requires_grad_(True)
    if exogenous is not None:
        exogenous = exogenous.to(device).clone().detach().requires_grad_(True)
    output = model(x, exogenous)
    output = output.sum()  # sum for batch-wise gradients
    output.backward()
    text_importance = x.grad.abs().mean(dim=-1)  # (batch_size, seq_len)
    result = {'text_importance': text_importance}
    if exogenous is not None and exogenous.grad is not None:
        exo_importance = exogenous.grad.abs()  # (batch_size, n_features)
        result['exogenous_importance'] = exo_importance
    
    # Restore original training state
    if not was_training:
        model.eval()
    
    return result

def get_attention_weights(model, x: torch.Tensor, exogenous: torch.Tensor) -> Optional[torch.Tensor]:
    """
    Extract attention weights from cross-attention layer (if available).
    Args:
        model: The textregress model
        x: Input tensor (batch_size, seq_len, features)
        exogenous: Exogenous features (batch_size, n_features)
    Returns:
        Attention weights tensor or None
    """
    if hasattr(model, 'cross_attention_enabled') and model.cross_attention_enabled:
        # Forward pass to get attention weights
        model.eval()
        # Ensure tensors are on the same device as the model
        device = next(model.parameters()).device
        x = x.to(device)
        exogenous = exogenous.to(device)
        with torch.no_grad():
            out, _ = model.rnn(x)
            global_token = torch.mean(out, dim=1)
            query = global_token.unsqueeze(1)
            exo_proj = model.cross_attention_exo_proj(exogenous)
            key_value = exo_proj.unsqueeze(1)
            _, attn_weights = model.cross_attention_layer(query, key_value, key_value, need_weights=True)
        return attn_weights  # (batch_size, num_heads, query_len, key_len)
    return None 