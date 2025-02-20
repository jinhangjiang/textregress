import torch
import torch.nn as nn
import pytorch_lightning as pl
from .loss import get_loss_function

class SqueezeExcitation(nn.Module):
    """
    A simple Squeeze-and-Excitation (SE) block for channel-wise recalibration.
    
    Args:
        channel (int): The number of input channels.
        reduction (int): Reduction ratio for the hidden layer. Default is 16.
    """
    def __init__(self, channel, reduction=16):
        super(SqueezeExcitation, self).__init__()
        self.fc1 = nn.Linear(channel, channel // reduction)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(channel // reduction, channel)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # x: (batch_size, channel)
        # Squeeze: Global average pooling over batch (per sample, average over feature dimension)
        se = x.mean(dim=0, keepdim=True)  # shape: (1, channel)
        se = self.fc1(se)
        se = self.relu(se)
        se = self.fc2(se)
        se = self.sigmoid(se)  # scale factor between 0 and 1
        return x * se  # broadcast multiplication

class TextRegressionModel(pl.LightningModule):
    """
    PyTorch Lightning model for text regression.
    
    Now with dropout layers applied after every major component and an optional
    squeeze-and-excitation (SE) block to enhance performance.
    """
    def __init__(self, 
                 rnn_type="LSTM", 
                 rnn_layers=2, 
                 hidden_size=512,
                 bidirectional=True, 
                 inference_layer_units=100, 
                 exogenous_features=None,
                 learning_rate=1e-3,
                 loss_function="mae",
                 encoder_output_dim=768,
                 optimizer_name="adam",
                 optimizer_params={},
                 cross_attention_enabled=False,
                 cross_attention_layer=None,
                 dropout_rate=0.0,
                 se_layer=True,
                 random_seed=1,
                 **kwargs):
        """
        Initialize the TextRegressionModel.
        
        Args:
            rnn_type (str): Type of RNN to use ("LSTM" or "GRU").
            rnn_layers (int): Number of RNN layers.
            hidden_size (int): Hidden size for the RNN.
            bidirectional (bool): Whether to use a bidirectional RNN.
            inference_layer_units (int): Number of units in the final inference layer.
            exogenous_features (list, optional): List of exogenous feature names.
            learning_rate (float): Learning rate for the optimizer.
            loss_function (str): Loss function to use. Options: "mae", "smape", "mse", "rmse", "wmape", "mape".
            encoder_output_dim (int): Dimensionality of the encoder's output.
            optimizer_name (str): Name of the optimizer (e.g., "adam", "sgd").
            optimizer_params (dict): Additional keyword arguments for the optimizer.
            cross_attention_enabled (bool): Enable cross attention between a global token and exogenous features.
            cross_attention_layer (nn.Module, optional): Custom cross attention layer. If not provided, a default
                nn.MultiheadAttention with one head is used.
            dropout_rate (float): Dropout rate to apply after every component. Default is 0.0.
            se_layer (bool): Whether to enable the squeeze-and-excitation block. Default is True.
            random_seed (int): Random seed for reproducibility. Default is 1.
        """
        super(TextRegressionModel, self).__init__()
        self.save_hyperparameters()
        
        # Set random seed for reproducibility.
        torch.manual_seed(random_seed)
        
        # RNN configuration.
        rnn_cls = nn.LSTM if rnn_type.upper() == "LSTM" else nn.GRU
        self.rnn = rnn_cls(
            input_size=encoder_output_dim,
            hidden_size=hidden_size,
            num_layers=rnn_layers,
            bidirectional=bidirectional,
            batch_first=True
        )
        self.rnn_output_dim = hidden_size * (2 if bidirectional else 1)
        
        # Standard inference layer (used when cross attention is disabled).
        self.inference = nn.Linear(self.rnn_output_dim, inference_layer_units)
        
        # Cross Attention mechanism.
        self.cross_attention_enabled = cross_attention_enabled
        if self.cross_attention_enabled:
            if cross_attention_layer is None:
                self.cross_attention_layer = nn.MultiheadAttention(embed_dim=self.rnn_output_dim, num_heads=1, batch_first=True)
            else:
                self.cross_attention_layer = cross_attention_layer
            # Project exogenous features to match rnn_output_dim.
            if exogenous_features is not None:
                self.cross_attention_exo_proj = nn.Linear(len(exogenous_features), self.rnn_output_dim)
            else:
                raise ValueError("cross_attention_enabled is True but exogenous_features is not provided.")
            # Inference layer for the concatenated vector (rnn output + cross attention output).
            self.inference_with_ca = nn.Linear(2 * self.rnn_output_dim, inference_layer_units)
        
        # Dropout layer (applied after every component).
        self.dropout = nn.Dropout(dropout_rate)
        
        # Squeeze-and-Excitation block.
        self.se_enabled = se_layer
        if self.se_enabled:
            self.se = SqueezeExcitation(inference_layer_units)
        
        # Final regressor.
        self.regressor = nn.Linear(inference_layer_units, 1)
        
        # Loss function.
        self.criterion = get_loss_function(loss_function)
        self.learning_rate = learning_rate
        
        self.optimizer_name = optimizer_name
        self.optimizer_params = optimizer_params
        
    def forward(self, x, exogenous=None):
        # RNN block.
        out, _ = self.rnn(x)  # out: (batch_size, seq_len, rnn_output_dim)
        rnn_last = out[:, -1, :]  # Last time step: (batch_size, rnn_output_dim)
        rnn_last = self.dropout(rnn_last)
        
        if self.cross_attention_enabled:
            # Generate a global token as the average over RNN outputs.
            global_token = torch.mean(out, dim=1)  # (batch_size, rnn_output_dim)
            global_token = self.dropout(global_token)
            query = global_token.unsqueeze(1)  # (batch_size, 1, rnn_output_dim)
            
            # Project exogenous features.
            exo_proj = self.cross_attention_exo_proj(exogenous)  # (batch_size, rnn_output_dim)
            exo_proj = self.dropout(exo_proj)
            key_value = exo_proj.unsqueeze(1)  # (batch_size, 1, rnn_output_dim)
            
            # Apply cross attention.
            cross_attn_out, _ = self.cross_attention_layer(query, key_value, key_value)
            cross_attn_out = cross_attn_out.squeeze(1)  # (batch_size, rnn_output_dim)
            cross_attn_out = self.dropout(cross_attn_out)
            
            # Concatenate RNN last output with cross attention output.
            combined = torch.cat([rnn_last, cross_attn_out], dim=1)  # (batch_size, 2*rnn_output_dim)
            inference_out = self.inference_with_ca(combined)
            inference_out = self.dropout(inference_out)
        else:
            inference_out = self.inference(rnn_last)
            inference_out = self.dropout(inference_out)
        
        if self.se_enabled:
            inference_out = self.se(inference_out)
            inference_out = self.dropout(inference_out)
        
        output = self.regressor(inference_out)
        return output
    
    def training_step(self, batch, batch_idx):
        if self.hparams.exogenous_features is not None:
            x, exogenous, y = batch
            y_hat = self(x, exogenous)
        else:
            x, y = batch
            y_hat = self(x)
        loss = self.criterion(y_hat.squeeze(), y.float())
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        if self.hparams.exogenous_features is not None:
            x, exogenous, y = batch
            y_hat = self(x, exogenous)
        else:
            x, y = batch
            y_hat = self(x)
        loss = self.criterion(y_hat.squeeze(), y.float())
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        if self.hparams.exogenous_features is not None:
            x, exogenous, _ = batch
            y_hat = self(x, exogenous)
        else:
            x, _ = batch
            y_hat = self(x)
        return y_hat.squeeze()
    
    def configure_optimizers(self):
        import torch.optim as optim
        optimizer_cls = None
        # Loop through all attributes in torch.optim
        for attr in dir(optim):
            if attr.lower() == self.optimizer_name.lower():
                optimizer_cls = getattr(optim, attr)
                break
        if optimizer_cls is None:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_name}")
        optimizer = optimizer_cls(self.parameters(), lr=self.learning_rate, **self.optimizer_params)
        return optimizer

