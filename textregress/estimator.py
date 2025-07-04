"""
Text regression estimator following an sklearn-like API.
"""

import math
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import random
import numpy as np

from .encoders import get_encoder
from .models import get_model, list_available_models
from .utils import chunk_text, pad_chunks, TextRegressionDataset, collate_fn

class TextRegressor:
    """
    A text regression estimator following an sklearn-like API.
    
    This estimator takes in a pandas DataFrame containing a 'text' column and a 'y'
    column (with optional exogenous feature columns) and processes the text using configurable
    encoding and chunking, then applies a deep learning model to predict the target variable.
    
    Additional encoder parameters can be passed via `encoder_params`. Also, the loss_function
    parameter can be provided as either a string (one of "mae", "mse", "rmse", "smape", "mape", "wmape")
    or as a custom callable loss function.
    """
    def __init__(self, 
                 model_name: str = "lstm",
                 encoder_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 encoder_params: dict = None,
                 chunk_info: tuple = None,
                 padding_value: int = 0,
                 exogenous_features: list = None,
                 learning_rate: float = 1e-3,
                 loss_function: str = "mae",
                 max_steps: int = 500,
                 early_stop_enabled: bool = False,
                 patience_steps: int = None,
                 val_check_steps: int = 50,
                 optimizer_name: str = "adam",
                 optimizer_params: dict = None,
                 random_seed: int = 1,
                 **model_params):
        """
        Initialize the TextRegressor.
        
        Args:
            model_name (str): Name of the model to use. Available models: {available_models}
            encoder_model (str): Pretrained encoder model identifier.
            encoder_params (dict, optional): Additional parameters to configure the encoder.
            chunk_info (tuple, optional): (chunk_size, overlap) for splitting long texts.
            padding_value (int, optional): Padding value for text chunks.
            exogenous_features (list, optional): List of additional exogenous feature column names.
            learning_rate (float): Learning rate for the optimizer.
            loss_function (str or callable): Loss function to use.
            max_steps (int): Maximum number of training steps.
            early_stop_enabled (bool): Whether to enable early stopping.
            patience_steps (int, optional): Number of steps with no improvement before stopping.
            val_check_steps (int): Interval for validation checks.
            optimizer_name (str): Name of the optimizer to use.
            optimizer_params (dict): Additional keyword arguments for the optimizer.
            random_seed (int): Random seed for reproducibility.
            **model_params: Additional model-specific parameters.
        """
        # Set random seed for reproducibility
        self.random_seed = random_seed
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        
        # Model configuration
        self.model_name = model_name
        self.model_params = model_params
        
        # Encoder configuration
        self.encoder_model = encoder_model
        self.encoder_params = encoder_params if encoder_params is not None else {}
        
        # Data processing configuration
        self.chunk_info = chunk_info
        self.padding_value = padding_value
        self.exogenous_features = exogenous_features
        
        # Training configuration
        self.learning_rate = learning_rate
        self.loss_function = loss_function
        self.max_steps = max_steps
        self.early_stop_enabled = early_stop_enabled
        self.val_check_steps = val_check_steps
        self.optimizer_name = optimizer_name
        self.optimizer_params = optimizer_params or {}
        
        if self.early_stop_enabled:
            self.patience_steps = patience_steps if patience_steps is not None else 10
        else:
            self.patience_steps = None
        
        # Initialize components
        self.encoder = get_encoder(self.encoder_model, **self.encoder_params)
        self.model = None
        self.exo_scaler = None

    def fit(self, df: pd.DataFrame, batch_size: int = 64, val_size: float = None, **kwargs) -> 'TextRegressor':
        """
        Fit the TextRegressor model on the provided DataFrame.
        
        Args:
            df (pandas.DataFrame): DataFrame containing 'text' and 'y' columns.
            batch_size (int): Batch size for training.
            val_size (float, optional): Proportion of data to use for validation.
            **kwargs: Additional arguments for model training.
            
        Returns:
            self: Fitted estimator.
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        if 'text' not in df.columns or 'y' not in df.columns:
            raise ValueError("DataFrame must have 'text' and 'y' columns")
        
        texts = df['text'].tolist()
        targets = df['y'].tolist()
        
        # Fit the encoder if necessary
        if hasattr(self.encoder, 'fitted') and not self.encoder.fitted:
            corpus = []
            for text in texts:
                if self.chunk_info:
                    max_length, overlap = self.chunk_info
                    chunks = chunk_text(text, max_length, overlap)
                else:
                    chunks = [text]
                chunks = pad_chunks(chunks, max_length if self.chunk_info else len(text), pad_token=" ")
                corpus.extend(chunks)
            self.encoder.fit(corpus)
        
        # Process texts
        encoded_sequences = []
        for text in tqdm(texts, desc="Processing texts"):
            if self.chunk_info:
                max_length, overlap = self.chunk_info
                chunks = chunk_text(text, max_length, overlap)
            else:
                chunks = [text]
            chunks = pad_chunks(chunks, max_length if self.chunk_info else len(text), pad_token=" ")
            encoded_chunks = [self.encoder.encode(chunk) for chunk in chunks]
            encoded_chunks = [chunk if isinstance(chunk, torch.Tensor) else torch.tensor(chunk)
                            for chunk in encoded_chunks]
            encoded_sequences.append(encoded_chunks)
        
        # Process exogenous features
        if self.exogenous_features is not None:
            exo_data = df[self.exogenous_features].values
            self.exo_scaler = StandardScaler()
            exo_data_scaled = self.exo_scaler.fit_transform(exo_data)
            exo_list = [list(row) for row in exo_data_scaled]
        else:
            exo_list = None
        
        # Create dataset
        dataset = TextRegressionDataset(encoded_sequences, targets, exogenous=exo_list)
        
        # Create data loaders
        if self.early_stop_enabled:
            if val_size is None:
                raise ValueError("When early_stop_enabled is True, you must specify val_size")
            indices = list(range(len(dataset)))
            train_idx, val_idx = train_test_split(indices, test_size=val_size, random_state=self.random_seed)
            train_subset = Subset(dataset, train_idx)
            val_subset = Subset(dataset, val_idx)
            train_loader = DataLoader(train_subset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)
        else:
            train_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
            val_loader = None
        
        # Calculate training parameters
        steps_per_epoch = len(train_loader)
        computed_epochs = math.ceil(self.max_steps / steps_per_epoch)
        
        # Get encoder output dimension
        if hasattr(self.encoder, 'model') and hasattr(self.encoder.model, 'get_sentence_embedding_dimension'):
            encoder_output_dim = self.encoder.model.get_sentence_embedding_dimension()
        elif hasattr(self.encoder, 'output_dim'):
            encoder_output_dim = self.encoder.output_dim
        else:
            encoder_output_dim = 768
        
        # Initialize model
        model_cls = get_model(self.model_name)
        self.model = model_cls(
            encoder_output_dim=encoder_output_dim,
            learning_rate=self.learning_rate,
            loss_function=self.loss_function,
            optimizer_name=self.optimizer_name,
            optimizer_params=self.optimizer_params,
            exogenous_features=self.exogenous_features,
            random_seed=self.random_seed,
            **self.model_params
        )
        
        # Configure callbacks
        callbacks = []
        if self.early_stop_enabled:
            from pytorch_lightning.callbacks import EarlyStopping
            early_stop_callback = EarlyStopping(
                monitor="val_loss",
                patience=self.patience_steps,
                mode="min",
                verbose=True,
            )
            callbacks.append(early_stop_callback)
        
        # Configure validation check interval
        if self.early_stop_enabled:
            val_check_interval = min(self.val_check_steps, len(train_loader))
        else:
            val_check_interval = None
        
        # Train model
        from pytorch_lightning import Trainer
        trainer = Trainer(
            max_steps=self.max_steps,
            max_epochs=computed_epochs,
            accelerator="auto",
            devices="auto",
            val_check_interval=val_check_interval,
            callbacks=callbacks
        )
        
        if self.early_stop_enabled:
            trainer.fit(self.model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        else:
            trainer.fit(self.model, train_dataloaders=train_loader)
        
        return self

    def predict(self, df: pd.DataFrame, batch_size: int = 64, **kwargs) -> np.ndarray:
        """
        Predict continuous target values for new text data.
        
        Args:
            df (pandas.DataFrame): DataFrame containing 'text' column and optional exogenous features.
            batch_size (int): Batch size for prediction.
            **kwargs: Additional arguments for prediction.
            
        Returns:
            np.ndarray: Predicted values.
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        if 'text' not in df.columns:
            raise ValueError("DataFrame must have a 'text' column")
        
        texts = df['text'].tolist()
        
        # Process texts
        encoded_sequences = []
        for text in tqdm(texts, desc="Processing texts"):
            if self.chunk_info:
                max_length, overlap = self.chunk_info
                chunks = chunk_text(text, max_length, overlap)
            else:
                chunks = [text]
            chunks = pad_chunks(chunks, max_length if self.chunk_info else len(text), pad_token=" ")
            encoded_chunks = [self.encoder.encode(chunk) for chunk in chunks]
            encoded_chunks = [chunk if isinstance(chunk, torch.Tensor) else torch.tensor(chunk)
                            for chunk in encoded_chunks]
            encoded_sequences.append(encoded_chunks)
        
        # Process exogenous features
        if self.exogenous_features is not None:
            exo_data = df[self.exogenous_features].values
            exo_data_scaled = self.exo_scaler.transform(exo_data)
            exo_list = [list(row) for row in exo_data_scaled]
        else:
            exo_list = None
        
        # Create dataset and dataloader
        dataset = TextRegressionDataset(encoded_sequences, [0] * len(texts), exogenous=exo_list)
        dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)
        
        # Make predictions
        from pytorch_lightning import Trainer
        trainer = Trainer(accelerator="auto", devices="auto")
        predictions = trainer.predict(self.model, dataloaders=dataloader)
        predictions = torch.cat(predictions).numpy()
        
        return predictions

    def fit_predict(self, df: pd.DataFrame, **kwargs) -> np.ndarray:
        """
        Fit the model and predict on the same data.
        
        Args:
            df (pandas.DataFrame): DataFrame containing 'text' and 'y' columns.
            **kwargs: Additional arguments for fit and predict.
            
        Returns:
            np.ndarray: Predicted values.
        """
        return self.fit(df, **kwargs).predict(df, **kwargs)
