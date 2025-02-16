import torch
import torch.nn as nn
import pytorch_lightning as pl
from .loss import get_loss_function

class TextRegressionModel(pl.LightningModule):
    """
    PyTorch Lightning model for text regression.
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
                 encoder_output_dim=768):
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
        """
        super(TextRegressionModel, self).__init__()
        self.save_hyperparameters()
        
        # Choose RNN class based on rnn_type.
        rnn_cls = nn.LSTM if rnn_type.upper() == "LSTM" else nn.GRU
        self.rnn = rnn_cls(
            input_size=encoder_output_dim,  # Dynamically set based on encoder.
            hidden_size=hidden_size,
            num_layers=rnn_layers,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Calculate RNN output dimension.
        rnn_output_dim = hidden_size * (2 if bidirectional else 1)
        
        # Define the inference block.
        self.inference = nn.Linear(rnn_output_dim, inference_layer_units)
        
        # If exogenous features are provided, adjust the regressor input dimension.
        if exogenous_features is not None:
            exo_dim = len(exogenous_features)
            self.regressor = nn.Linear(inference_layer_units + exo_dim, 1)
        else:
            self.regressor = nn.Linear(inference_layer_units, 1)
        
        # Set up the loss function.
        self.criterion = get_loss_function(loss_function)
        self.learning_rate = learning_rate
        
    def forward(self, x, exogenous=None):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Encoded text data of shape (batch_size, seq_len, feature_dim).
            exogenous (torch.Tensor, optional): Exogenous features.
            
        Returns:
            torch.Tensor: Regression output.
        """
        out, _ = self.rnn(x)
        # Use the output from the last time step.
        out = out[:, -1, :]
        out = self.inference(out)
        
        if exogenous is not None:
            out = torch.cat([out, exogenous], dim=1)
        
        output = self.regressor(out)
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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
