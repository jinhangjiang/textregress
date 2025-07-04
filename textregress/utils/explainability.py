"""
Lightweight explainability utilities for textregress models.

Provides:
- Gradient-based feature importance (saliency)
- Integrated gradients (optional, efficient)
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
    model.eval()
    x = x.clone().detach().requires_grad_(True)
    if exogenous is not None:
        exogenous = exogenous.clone().detach().requires_grad_(True)
    output = model(x, exogenous)
    output = output.sum()  # sum for batch-wise gradients
    output.backward()
    text_importance = x.grad.abs().mean(dim=-1)  # (batch_size, seq_len)
    result = {'text_importance': text_importance}
    if exogenous is not None and exogenous.grad is not None:
        exo_importance = exogenous.grad.abs()  # (batch_size, n_features)
        result['exogenous_importance'] = exo_importance
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
        with torch.no_grad():
            out, _ = model.rnn(x)
            global_token = torch.mean(out, dim=1)
            query = global_token.unsqueeze(1)
            exo_proj = model.cross_attention_exo_proj(exogenous)
            key_value = exo_proj.unsqueeze(1)
            _, attn_weights = model.cross_attention_layer(query, key_value, key_value, need_weights=True)
        return attn_weights  # (batch_size, num_heads, query_len, key_len)
    return None

def integrated_gradients(model, x: torch.Tensor, exogenous: Optional[torch.Tensor] = None, baseline: Optional[torch.Tensor] = None, steps: int = 20) -> Dict[str, torch.Tensor]:
    """
    Compute integrated gradients for input text and exogenous features.
    Args:
        model: The textregress model
        x: Input tensor (batch_size, seq_len, features)
        exogenous: Exogenous features (batch_size, n_features) or None
        baseline: Baseline tensor (same shape as x) or None (defaults to zeros)
        steps: Number of steps for integration
    Returns:
        Dict with 'text_importance' and optionally 'exogenous_importance'
    """
    if baseline is None:
        baseline = torch.zeros_like(x)
    scaled_inputs = [baseline + (float(i) / steps) * (x - baseline) for i in range(steps + 1)]
    grads = []
    for scaled_x in scaled_inputs:
        scaled_x = scaled_x.clone().detach().requires_grad_(True)
        if exogenous is not None:
            exo = exogenous.clone().detach().requires_grad_(True)
        else:
            exo = None
        output = model(scaled_x, exo)
        output = output.sum()
        output.backward()
        grads.append(scaled_x.grad.detach().clone())
    avg_grads = torch.stack(grads).mean(dim=0)
    text_importance = (x - baseline) * avg_grads
    result = {'text_importance': text_importance.abs().mean(dim=-1)}
    if exogenous is not None:
        # Integrated gradients for exogenous features
        exo_grads = []
        for scaled_x in scaled_inputs:
            scaled_x = scaled_x.clone().detach().requires_grad_(True)
            exo = exogenous.clone().detach().requires_grad_(True)
            output = model(scaled_x, exo)
            output = output.sum()
            output.backward()
            exo_grads.append(exo.grad.detach().clone())
        avg_exo_grads = torch.stack(exo_grads).mean(dim=0)
        exo_importance = exogenous * avg_exo_grads
        result['exogenous_importance'] = exo_importance.abs()
    return result 