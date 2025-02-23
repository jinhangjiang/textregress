import torch

def mae_loss(pred, target):
    """Mean Absolute Error"""
    return torch.mean(torch.abs(pred - target))

def mse_loss(pred, target):
    """Mean Squared Error"""
    return torch.mean((pred - target) ** 2)

def rmse_loss(pred, target):
    """Root Mean Squared Error"""
    return torch.sqrt(mse_loss(pred, target) + 1e-8)

def smape_loss(pred, target):
    """Symmetric Mean Absolute Percentage Error"""
    denominator = (torch.abs(pred) + torch.abs(target)) / 2.0 + 1e-8
    return torch.mean(torch.abs(pred - target) / denominator)

def mape_loss(pred, target):
    """Mean Absolute Percentage Error"""
    return torch.mean(torch.abs((target - pred) / (target + 1e-8))) * 100.0

def wmape_loss(pred, target):
    """Weighted Mean Absolute Percentage Error"""
    return torch.sum(torch.abs(target - pred)) / (torch.sum(torch.abs(target)) + 1e-8)

def get_loss_function(loss_name):
    """
    Factory function to get the loss function.
    
    Args:
        loss_name (str or callable): If a string, one of "mae", "mse", "rmse", "smape", "mape", "wmape".
            If a callable is provided, it will be returned directly.
        
    Returns:
        A callable loss function.
    """
    if callable(loss_name):
        return loss_name
    loss_name = loss_name.lower()
    if loss_name == 'mae':
        return mae_loss
    elif loss_name == 'mse':
        return mse_loss
    elif loss_name == 'rmse':
        return rmse_loss
    elif loss_name == 'smape':
        return smape_loss
    elif loss_name == 'mape':
        return mape_loss
    elif loss_name == 'wmape':
        return wmape_loss
    else:
        raise ValueError(f"Unsupported loss function: {loss_name}")
