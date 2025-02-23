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
        # Squeeze: Global average pooling over feature dimension.
        se = x.mean(dim=0, keepdim=True)  # shape: (1, channel)
        se = self.fc1(se)
        se = self.relu(se)
        se = self.fc2(se)
        se = self.sigmoid(se)  # scale factors between 0 and 1
        return x * se  # broadcast multiplication

class TextRegressionModel(pl.LightningModule):
    """
    PyTorch Lightning model for text regression.
    
    This model now handles exogenous features in two ways:
    
    1) When cross_attention_enabled is False but exogenous_features are provided, 
       the normalized exogenous features are concatenated with the document embedding.
       
    2) When the additional parameter feature_mixer is True, the model first computes an 
       inference output from the document embedding and then concatenates normalized exogenous 
       features with that output before passing through a dedicated mixing layer.
       
    In both cases, a simple LayerNorm is applied to the exogenous features.
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
                 encoder_output_dim=None,
                 optimizer_name="adam",
                 optimizer_params={},
                 cross_attention_enabled=False,
                 cross_attention_layer=None,
                 dropout_rate=0.0,
                 se_layer=True,
                 feature_mixer=False,
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
            encoder_output_dim (int): Dimensionality of the encoder's output. **Must be provided.**
            optimizer_name (str): Name of the optimizer (e.g., "adam", "sgd").
            optimizer_params (dict): Additional keyword arguments for the optimizer.
            cross_attention_enabled (bool): Enable cross attention between a global token and exogenous features.
            cross_attention_layer (nn.Module, optional): Custom cross attention layer.
            dropout_rate (float): Dropout rate applied after each component.
            se_layer (bool): Whether to enable the squeeze-and-excitation block.
            feature_mixer (bool): If True, mix normalized exogenous features with the document embedding's inference output.
            random_seed (int): Random seed for reproducibility.
            **kwargs: Additional keyword arguments.
        """
        super(TextRegressionModel, self).__init__()
        self.save_hyperparameters()
        
        if encoder_output_dim is None:
            raise ValueError("encoder_output_dim must be provided and match the encoder's output dimension.")
        
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
        
        # Branch for Cross Attention.
        self.cross_attention_enabled = cross_attention_enabled
        if self.cross_attention_enabled:
            if cross_attention_layer is None:
                self.cross_attention_layer = nn.MultiheadAttention(embed_dim=self.rnn_output_dim, num_heads=1, batch_first=True)
            else:
                self.cross_attention_layer = cross_attention_layer
            if exogenous_features is not None:
                self.cross_attention_exo_proj = nn.Linear(len(exogenous_features), self.rnn_output_dim)
            else:
                raise ValueError("cross_attention_enabled is True but exogenous_features is not provided.")
            # Inference layer for the concatenated vector (document embedding + cross attention output).
            self.inference_with_ca = nn.Linear(2 * self.rnn_output_dim, inference_layer_units)
        # Branch for non-cross attention but with exogenous features.
        elif exogenous_features is not None:
            self.exo_dim = len(exogenous_features)
            self.exo_norm = nn.LayerNorm(self.exo_dim)
            if feature_mixer:
                # First, compute an inference output from the document embedding.
                self.inference = nn.Linear(self.rnn_output_dim, inference_layer_units)
                # Then, mix in the exogenous features.
                self.feature_mixer_layer = nn.Linear(inference_layer_units + self.exo_dim, inference_layer_units)
            else:
                # Directly concatenate document embedding and exogenous features.
                self.inference = nn.Linear(self.rnn_output_dim + self.exo_dim, inference_layer_units)
        else:
            # No exogenous features.
            self.inference = nn.Linear(self.rnn_output_dim, inference_layer_units)
        
        # Dropout layer applied after every component.
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
        
        if self.hparams.cross_attention_enabled:
            # Cross attention branch.
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
            # Non-cross attention branch.
            if exogenous is not None:
                if self.hparams.feature_mixer:
                    # Compute inference output from document embedding only.
                    inference_out = self.inference(rnn_last)
                    inference_out = self.dropout(inference_out)
                    # Normalize exogenous features.
                    exo = self.exo_norm(exogenous)
                    # Concatenate and mix.
                    combined = torch.cat([inference_out, exo], dim=1)
                    inference_out = self.feature_mixer_layer(combined)
                    inference_out = self.dropout(inference_out)
                else:
                    # Directly concatenate document embedding and normalized exogenous features.
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
        
        output = self.regressor(inference_out)
        return output
    
    def training_step(self, batch, batch_idx):
        if self.hparams.exogenous_features is not None:
            x, exogenous, y = batch
            y_hat = self(x, exogenous)
        else:
            x, y = batch
            y_hat = self(x)
        # Call the loss function using keyword arguments so that custom losses expecting "pred" and "target"
        # pick them up automatically.
        loss = self.criterion(pred=y_hat.squeeze(), target=y.float())
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        if self.hparams.exogenous_features is not None:
            x, exogenous, y = batch
            y_hat = self(x, exogenous)
        else:
            x, y = batch
            y_hat = self(x)
        loss = self.criterion(pred=y_hat.squeeze(), target=y.float())
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
        # Loop through torch.optim to find the matching optimizer.
        for attr in dir(optim):
            if attr.lower() == self.optimizer_name.lower():
                optimizer_cls = getattr(optim, attr)
                break
        if optimizer_cls is None:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_name}")
        optimizer = optimizer_cls(self.parameters(), lr=self.learning_rate, **self.optimizer_params)
        return optimizer
