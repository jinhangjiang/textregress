"""
Base model interface for textregress.

This module defines the base class that all text regression models must implement.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Union, Tuple
import torch
import pytorch_lightning as pl
from ..losses import get_loss_function
from ..utils.explainability import get_gradient_importance, get_attention_weights, integrated_gradients

class BaseTextRegressionModel(pl.LightningModule, ABC):
    """
    Abstract base class for all text regression models.
    
    This class defines the interface that all text regression models must implement.
    It inherits from PyTorch Lightning's LightningModule to provide training functionality.
    """
    
    def __init__(self,
                 encoder_output_dim: int,
                 learning_rate: float = 1e-3,
                 loss_function: str = "mae",
                 optimizer_name: str = "adam",
                 optimizer_params: Optional[Dict[str, Any]] = None,
                 random_seed: int = 1,
                 **kwargs):
        """
        Initialize the base model.
        
        Args:
            encoder_output_dim (int): Dimensionality of the encoder's output.
            learning_rate (float): Learning rate for the optimizer.
            loss_function (str): Loss function to use.
            optimizer_name (str): Name of the optimizer to use.
            optimizer_params (Dict[str, Any], optional): Additional optimizer parameters.
            random_seed (int): Random seed for reproducibility.
            **kwargs: Additional model-specific parameters.
        """
        super().__init__()
        self.save_hyperparameters()
        
        # Set random seed for reproducibility
        torch.manual_seed(random_seed)
        
        # Store configuration parameters
        self.encoder_output_dim = encoder_output_dim
        self.loss_function = loss_function
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer_name
        self.optimizer_params = optimizer_params or {}
        self.random_seed = random_seed
        
        # Initialize loss function
        loss_cls = get_loss_function(loss_function)
        self.criterion = loss_cls()
    
    @abstractmethod
    def forward(self, x: torch.Tensor, exogenous: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, features).
            exogenous (torch.Tensor, optional): Exogenous features of shape (batch_size, n_features).
            
        Returns:
            torch.Tensor: Model predictions of shape (batch_size, 1).
        """
        pass
    
    @abstractmethod
    def get_document_embedding(self, x: torch.Tensor, exogenous: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Get the document embedding from the model.
        
        This method should return the document representation before the final regression layer.
        This is useful for transfer learning, feature extraction, or analysis.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, features).
            exogenous (torch.Tensor, optional): Exogenous features of shape (batch_size, n_features).
            
        Returns:
            torch.Tensor: Document embeddings of shape (batch_size, embedding_dim).
        """
        pass
    
    @abstractmethod
    def get_sequence_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get the sequence embeddings from the model.
        
        This method should return the sequence-level representations (e.g., from RNN layers)
        before any pooling or aggregation.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, features).
            
        Returns:
            torch.Tensor: Sequence embeddings of shape (batch_size, sequence_length, hidden_dim).
        """
        pass
    
    def get_gradient_importance(self, x: torch.Tensor, exogenous: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute gradient-based feature importance for input text and exogenous features.
        """
        return get_gradient_importance(self, x, exogenous)
    
    def get_attention_weights(self, x: torch.Tensor, exogenous: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Extract attention weights from cross-attention layer (if available).
        """
        return get_attention_weights(self, x, exogenous)
    
    def integrated_gradients(self, x: torch.Tensor, exogenous: Optional[torch.Tensor] = None, baseline: Optional[torch.Tensor] = None, steps: int = 20) -> Dict[str, torch.Tensor]:
        """
        Compute integrated gradients for input text and exogenous features.
        """
        return integrated_gradients(self, x, exogenous, baseline, steps)
    
    def training_step(self, batch: Union[tuple, dict], batch_idx: int) -> torch.Tensor:
        """
        Training step.
        
        Args:
            batch: Either (x, y), (x, exogenous, y), or dict with keys 'x', 'y', 'exogenous'.
            batch_idx: Index of the current batch.
            
        Returns:
            torch.Tensor: Loss value.
        """
        if isinstance(batch, dict):
            x = batch['x']
            y = batch['y']
            exogenous = batch.get('exogenous', None)
        elif len(batch) == 3:
            x, exogenous, y = batch
        else:
            x, y = batch
            exogenous = None
        
        y_hat = self(x, exogenous)
        loss = self.criterion(y_hat.squeeze(), y.float())
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch: Union[tuple, dict], batch_idx: int) -> torch.Tensor:
        """
        Validation step.
        
        Args:
            batch: Either (x, y), (x, exogenous, y), or dict with keys 'x', 'y', 'exogenous'.
            batch_idx: Index of the current batch.
            
        Returns:
            torch.Tensor: Loss value.
        """
        if isinstance(batch, dict):
            x = batch['x']
            y = batch['y']
            exogenous = batch.get('exogenous', None)
        elif len(batch) == 3:
            x, exogenous, y = batch
        else:
            x, y = batch
            exogenous = None
        
        y_hat = self(x, exogenous)
        loss = self.criterion(y_hat.squeeze(), y.float())
        self.log('val_loss', loss, prog_bar=True)
        return loss
    
    def predict_step(self, batch: Union[tuple, dict], batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        """
        Prediction step.
        
        Args:
            batch: Either (x, _), (x, exogenous, _), or dict with keys 'x', 'y', 'exogenous'.
            batch_idx: Index of the current batch.
            dataloader_idx: Index of the current dataloader.
            
        Returns:
            torch.Tensor: Model predictions.
        """
        if isinstance(batch, dict):
            x = batch['x']
            exogenous = batch.get('exogenous', None)
        elif len(batch) == 3:
            x, exogenous, _ = batch
        else:
            x, _ = batch
            exogenous = None
        
        y_hat = self(x, exogenous)
        return y_hat.squeeze()
    
    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Configure the optimizer.
        
        Returns:
            torch.optim.Optimizer: The configured optimizer.
        """
        import torch.optim as optim
        
        # Find the optimizer class
        optimizer_cls = None
        for attr in dir(optim):
            if attr.lower() == self.optimizer_name.lower():
                optimizer_cls = getattr(optim, attr)
                break
        
        if optimizer_cls is None:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_name}")
        
        return optimizer_cls(self.parameters(), lr=self.learning_rate, **self.optimizer_params) 