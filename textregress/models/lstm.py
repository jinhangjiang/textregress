"""
LSTM-based text regression model implementation.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List
from .base import BaseTextRegressionModel
from .registry import register_model

class SqueezeExcitation(nn.Module):
    """
    A simple Squeeze-and-Excitation (SE) block for channel-wise recalibration.
    
    Args:
        channel (int): The number of input channels.
        reduction (int): Reduction ratio for the hidden layer. Default is 16.
    """
    def __init__(self, channel: int, reduction: int = 16):
        super(SqueezeExcitation, self).__init__()
        self.fc1 = nn.Linear(channel, channel // reduction)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(channel // reduction, channel)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, channel)
        # Squeeze: Global average pooling over feature dimension.
        se = x.mean(dim=0, keepdim=True)  # shape: (1, channel)
        se = self.fc1(se)
        se = self.relu(se)
        se = self.fc2(se)
        se = self.sigmoid(se)  # scale factors between 0 and 1
        return x * se  # broadcast multiplication

@register_model("lstm")
class LSTMTextRegressionModel(BaseTextRegressionModel):
    """
    LSTM-based text regression model.
    
    This model uses an LSTM network to process text sequences and can optionally
    incorporate exogenous features through cross-attention or direct concatenation.
    """
    
    def __init__(self,
                 rnn_type: str = "LSTM",
                 rnn_layers: int = 2,
                 hidden_size: int = 512,
                 bidirectional: bool = True,
                 inference_layer_units: int = 100,
                 exogenous_features: Optional[List[str]] = None,
                 learning_rate: float = 1e-3,
                 loss_function: str = "mae",
                 encoder_output_dim: int = 768,
                 optimizer_name: str = "adam",
                 optimizer_params: Optional[Dict[str, Any]] = None,
                 cross_attention_enabled: bool = False,
                 cross_attention_layer: Optional[nn.Module] = None,
                 dropout_rate: float = 0.0,
                 se_layer: bool = True,
                 feature_mixer: bool = False,
                 random_seed: int = 1,
                 **kwargs):
        """
        Initialize the LSTM text regression model.
        
        Args:
            rnn_type (str): Type of RNN to use ("LSTM" or "GRU").
            rnn_layers (int): Number of RNN layers.
            hidden_size (int): Hidden size for the RNN.
            bidirectional (bool): Whether to use bidirectional RNN.
            inference_layer_units (int): Number of units in the final inference layer.
            exogenous_features (List[str], optional): List of exogenous feature names.
            learning_rate (float): Learning rate for the optimizer.
            loss_function (str): Loss function to use.
            encoder_output_dim (int): Dimensionality of the encoder's output.
            optimizer_name (str): Name of the optimizer to use.
            optimizer_params (Dict[str, Any], optional): Additional optimizer parameters.
            cross_attention_enabled (bool): Whether to enable cross attention.
            cross_attention_layer (nn.Module, optional): Custom cross attention layer.
            dropout_rate (float): Dropout rate to apply.
            se_layer (bool): Whether to enable the squeeze-and-excitation block.
            feature_mixer (bool): Whether to use feature mixing for exogenous features.
            random_seed (int): Random seed for reproducibility.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(
            encoder_output_dim=encoder_output_dim,
            learning_rate=learning_rate,
            loss_function=loss_function,
            optimizer_name=optimizer_name,
            optimizer_params=optimizer_params,
            random_seed=random_seed,
            **kwargs
        )
        
        # Store configuration parameters
        self.rnn_type = rnn_type
        self.rnn_layers = rnn_layers
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.inference_layer_units = inference_layer_units
        self.exogenous_features = exogenous_features
        self.cross_attention_enabled = cross_attention_enabled
        self.dropout_rate = dropout_rate
        self.se_enabled = se_layer
        self.feature_mixer = feature_mixer
        
        # RNN configuration
        rnn_cls = nn.LSTM if rnn_type.upper() == "LSTM" else nn.GRU
        self.rnn = rnn_cls(
            input_size=encoder_output_dim,
            hidden_size=hidden_size,
            num_layers=rnn_layers,
            bidirectional=bidirectional,
            batch_first=True
        )
        self.rnn_output_dim = hidden_size * (2 if bidirectional else 1)
        
        # Cross attention configuration
        self.cross_attention_enabled = cross_attention_enabled
        if self.cross_attention_enabled:
            if cross_attention_layer is None:
                self.cross_attention_layer = nn.MultiheadAttention(
                    embed_dim=self.rnn_output_dim,
                    num_heads=1,
                    batch_first=True
                )
            else:
                self.cross_attention_layer = cross_attention_layer
            if exogenous_features is not None:
                self.cross_attention_exo_proj = nn.Linear(len(exogenous_features), self.rnn_output_dim)
            else:
                raise ValueError("cross_attention_enabled is True but exogenous_features is not provided.")
            self.inference_with_ca = nn.Linear(2 * self.rnn_output_dim, inference_layer_units)
        
        # Exogenous features configuration
        elif exogenous_features is not None:
            self.exo_dim = len(exogenous_features)
            self.exo_norm = nn.LayerNorm(self.exo_dim)
            if feature_mixer:
                self.inference = nn.Linear(self.rnn_output_dim, inference_layer_units)
                self.feature_mixer_layer = nn.Linear(inference_layer_units + self.exo_dim, inference_layer_units)
            else:
                self.inference = nn.Linear(self.rnn_output_dim + self.exo_dim, inference_layer_units)
        else:
            self.inference = nn.Linear(self.rnn_output_dim, inference_layer_units)
        
        # Additional layers
        self.dropout = nn.Dropout(dropout_rate)
        self.se_enabled = se_layer
        if self.se_enabled:
            self.se = SqueezeExcitation(inference_layer_units)
        self.regressor = nn.Linear(inference_layer_units, 1)
    
    def get_sequence_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get the sequence embeddings from the LSTM layers.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, features).
            
        Returns:
            torch.Tensor: Sequence embeddings of shape (batch_size, sequence_length, rnn_output_dim).
        """
        out, _ = self.rnn(x)  # out: (batch_size, seq_len, rnn_output_dim)
        return out
    
    def get_document_embedding(self, x: torch.Tensor, exogenous: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Get the document embedding from the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, features).
            exogenous (torch.Tensor, optional): Exogenous features of shape (batch_size, n_features).
            
        Returns:
            torch.Tensor: Document embeddings of shape (batch_size, inference_layer_units).
        """
        # RNN block
        out, _ = self.rnn(x)  # out: (batch_size, seq_len, rnn_output_dim)
        rnn_last = out[:, -1, :]  # Last time step: (batch_size, rnn_output_dim)
        rnn_last = self.dropout(rnn_last)
        
        if self.cross_attention_enabled:
            # Cross attention branch
            global_token = torch.mean(out, dim=1)  # (batch_size, rnn_output_dim)
            global_token = self.dropout(global_token)
            query = global_token.unsqueeze(1)  # (batch_size, 1, rnn_output_dim)
            
            exo_proj = self.cross_attention_exo_proj(exogenous)  # (batch_size, rnn_output_dim)
            exo_proj = self.dropout(exo_proj)
            key_value = exo_proj.unsqueeze(1)  # (batch_size, 1, rnn_output_dim)
            
            cross_attn_out, _ = self.cross_attention_layer(query, key_value, key_value)
            cross_attn_out = cross_attn_out.squeeze(1)  # (batch_size, rnn_output_dim)
            cross_attn_out = self.dropout(cross_attn_out)
            
            combined = torch.cat([rnn_last, cross_attn_out], dim=1)  # (batch_size, 2*rnn_output_dim)
            inference_out = self.inference_with_ca(combined)
            inference_out = self.dropout(inference_out)
        else:
            # Non-cross attention branch
            if exogenous is not None:
                if hasattr(self, 'feature_mixer_layer'):
                    # Compute inference output from document embedding only
                    inference_out = self.inference(rnn_last)
                    inference_out = self.dropout(inference_out)
                    # Normalize exogenous features
                    exo = self.exo_norm(exogenous)
                    # Concatenate and mix
                    combined = torch.cat([inference_out, exo], dim=1)
                    inference_out = self.feature_mixer_layer(combined)
                    inference_out = self.dropout(inference_out)
                else:
                    # Directly concatenate document embedding and normalized exogenous features
                    exo = self.exo_norm(exogenous)
                    combined = torch.cat([rnn_last, exo], dim=1)
                    inference_out = self.inference(combined)
                    inference_out = self.dropout(inference_out)
            else:
                inference_out = self.inference(rnn_last)
                inference_out = self.dropout(inference_out)
        
        if self.se_enabled:
            inference_out = self.se(inference_out)
            inference_out = self.dropout(inference_out)
        
        return inference_out
    
    def forward(self, x: torch.Tensor, exogenous: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, features).
            exogenous (torch.Tensor, optional): Exogenous features of shape (batch_size, n_features).
            
        Returns:
            torch.Tensor: Model predictions of shape (batch_size, 1).
        """
        document_embedding = self.get_document_embedding(x, exogenous)
        output = self.regressor(document_embedding)
        return output 

    def fit_predict(self, train_loader, val_loader=None, max_epochs=10, optimizer=None, criterion=None, device=None):
        """
        Fit the model and return predictions on the training set.
        Args:
            train_loader: DataLoader for training data
            val_loader: Optional DataLoader for validation
            max_epochs: Number of epochs
            optimizer: Optional optimizer (if None, uses self.configure_optimizers())
            criterion: Optional loss function (if None, uses self.criterion)
            device: Optional device (if None, uses 'cuda' if available)
        Returns:
            predictions: Model predictions on the training set
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        if optimizer is None:
            optimizer = self.configure_optimizers()
        if criterion is None:
            criterion = self.criterion
        self.train()
        for epoch in range(max_epochs):
            for batch in train_loader:
                optimizer.zero_grad()
                if len(batch) == 3:
                    x, exogenous, y = batch
                    x, exogenous, y = x.to(device), exogenous.to(device), y.to(device)
                    y_hat = self(x, exogenous)
                else:
                    x, y = batch
                    x, y = x.to(device), y.to(device)
                    y_hat = self(x)
                loss = criterion(y_hat.squeeze(), y.float())
                loss.backward()
                optimizer.step()
        self.eval()
        predictions = []
        with torch.no_grad():
            for batch in train_loader:
                if len(batch) == 3:
                    x, exogenous, _ = batch
                    x, exogenous = x.to(device), exogenous.to(device)
                    y_hat = self(x, exogenous)
                else:
                    x, _ = batch
                    x = x.to(device)
                    y_hat = self(x)
                predictions.append(y_hat.cpu())
        return torch.cat(predictions)

    def save(self, path: str):
        """
        Save the model to a file.
        Args:
            path: File path to save the model
        """
        # Save both state dict and model configuration
        save_dict = {
            'state_dict': self.state_dict(),
            'model_config': {
                'rnn_type': self.rnn_type,
                'rnn_layers': self.rnn_layers,
                'hidden_size': self.hidden_size,
                'bidirectional': self.bidirectional,
                'inference_layer_units': self.inference_layer_units,
                'exogenous_features': self.exogenous_features,
                'learning_rate': self.learning_rate,
                'loss_function': self.loss_function,
                'encoder_output_dim': self.encoder_output_dim,
                'optimizer_name': self.optimizer_name,
                'optimizer_params': self.optimizer_params,
                'cross_attention_enabled': self.cross_attention_enabled,
                'dropout_rate': self.dropout_rate,
                'se_layer': self.se_enabled,
                'feature_mixer': hasattr(self, 'feature_mixer_layer'),
                'random_seed': self.random_seed
            }
        }
        torch.save(save_dict, path)

    @classmethod
    def load(cls, path: str):
        """
        Load a model from a file.
        Args:
            path: File path to load the model from
        Returns:
            model: Loaded model instance
        """
        save_dict = torch.load(path, map_location='cpu')
        model_config = save_dict['model_config']
        model = cls(**model_config)
        model.load_state_dict(save_dict['state_dict'])
        return model 