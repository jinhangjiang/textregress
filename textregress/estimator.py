"""
Text regression estimator following an sklearn-like API.
"""

import math
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Subset
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import random
import numpy as np

from .encoders import get_encoder
from .models import get_model, list_available_models
from .utils import chunk_text, pad_chunks, TextRegressionDataset, collate_fn


def _setup_device(device: str) -> torch.device:
    """
    Setup device for model and tensors.
    
    Args:
        device (str): Device specification. Options:
            - "auto": Automatically select best available device
            - "cpu": Force CPU usage
            - "cuda": Force CUDA usage (if available)
            - "cuda:0", "cuda:1", etc.: Specific CUDA device
            - torch.device object: Direct device object
            
    Returns:
        torch.device: The configured device
    """
    if isinstance(device, torch.device):
        return device
    
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    elif device == "cpu":
        return torch.device("cpu")
    elif device.startswith("cuda"):
        if torch.cuda.is_available():
            return torch.device(device)
        else:
            print(f"Warning: CUDA requested but not available. Falling back to CPU.")
            return torch.device("cpu")
    else:
        try:
            return torch.device(device)
        except:
            print(f"Warning: Invalid device '{device}'. Falling back to auto.")
            return _setup_device("auto")


def _ensure_tensor_on_device(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Ensure tensor is on the specified device.
    
    Args:
        tensor (torch.Tensor): Input tensor
        device (torch.device): Target device
        
    Returns:
        torch.Tensor: Tensor on the target device
    """
    if tensor.device != device:
        return tensor.to(device)
    return tensor


def _ensure_model_on_device(model, device: torch.device):
    """
    Ensure model is on the specified device.
    
    Args:
        model: PyTorch model
        device (torch.device): Target device
    """
    if next(model.parameters()).device != device:
        model.to(device)

class TextRegressor:
    """
    A text regression estimator following an sklearn-like API.
    
    This estimator takes in a pandas DataFrame containing a 'text' column and a 'y'
    column (with optional exogenous feature columns) and processes the text using configurable
    encoding and chunking, then applies a deep learning model to predict the target variable.
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
                 device: str = "auto",
                 # Model hyperparameters
                 rnn_layers: int = 2,
                 hidden_size: int = 512,
                 bidirectional: bool = True,
                 inference_layer_units: int = 100,
                 cross_attention_enabled: bool = False,
                 cross_attention_layer: object = None,
                 dropout_rate: float = 0.0,
                 se_layer: bool = True,
                 feature_mixer: bool = False,
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
            rnn_layers (int): Number of RNN layers.
            hidden_size (int): Hidden size for the RNN.
            bidirectional (bool): Whether to use bidirectional RNN.
            inference_layer_units (int): Number of units in the final inference layer.
            cross_attention_enabled (bool): Whether to enable cross attention.
            cross_attention_layer (object): Custom cross attention layer.
            dropout_rate (float): Dropout rate to apply.
            se_layer (bool): Whether to enable the squeeze-and-excitation block.
            feature_mixer (bool): Whether to use feature mixing for exogenous features.
            **model_params: Additional model-specific parameters.
        """
        # Set random seed for reproducibility
        self.random_seed = random_seed
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        
        # Device configuration
        self.device = self._setup_device(device)
        
        # Model configuration
        self.model_name = model_name
        self.model_params = dict(
            rnn_layers=rnn_layers,
            hidden_size=hidden_size,
            bidirectional=bidirectional,
            inference_layer_units=inference_layer_units,
            cross_attention_enabled=cross_attention_enabled,
            cross_attention_layer=cross_attention_layer,
            dropout_rate=dropout_rate,
            se_layer=se_layer,
            feature_mixer=feature_mixer,
            **model_params
        )
        
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
        # Determine encoder type from model name
        if self.encoder_model.startswith("sentence-transformers/"):
            encoder_type = "sentence_transformer"
        elif self.encoder_model == "tfidf":
            encoder_type = "tfidf"
        else:
            encoder_type = "sentence_transformer"  # Default fallback
        
        # Pass the model name as a parameter to the encoder (only for sentence transformers)
        encoder_params = self.encoder_params.copy() if self.encoder_params else {}
        if encoder_type == "sentence_transformer":
            encoder_params['model_name'] = self.encoder_model
        
        self.encoder = get_encoder(encoder_type, **encoder_params)
        self.model = None
        self.exo_scaler = None

    def _setup_device(self, device: str) -> torch.device:
        """
        Setup device for model and tensors.
        
        Args:
            device (str): Device specification
            
        Returns:
            torch.device: The configured device
        """
        return _setup_device(device)

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
        
        # Store training data for feature importance analysis
        self._training_data = df.copy()
        
        texts = df['text'].tolist()
        targets = df['y'].tolist()
        
        # Fit the encoder if necessary
        if hasattr(self.encoder, 'fitted') and not self.encoder.fitted:
            corpus = []
            for i, text in enumerate(texts):
                if self.chunk_info:
                    max_length, overlap = self.chunk_info
                    chunks = chunk_text(text, max_length, overlap)
                else:
                    chunks = [text]
                chunks = pad_chunks(chunks, self.padding_value)
                corpus.extend(chunks)
            
            # For TF-IDF, we fit on all chunks to learn vocabulary and IDF
            # For other encoders, this might be a no-op or minimal setup
            self.encoder.fit(corpus)
        
        # Process texts
        encoded_sequences = []
        for i, text in enumerate(tqdm(texts, desc="Processing texts")):
            if self.chunk_info:
                max_length, overlap = self.chunk_info
                chunks = chunk_text(text, max_length, overlap)
            else:
                chunks = [text]
            chunks = pad_chunks(chunks, self.padding_value)
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
        
        # Ensure model is on the correct device
        _ensure_model_on_device(self.model, self.device)
        
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
            chunks = pad_chunks(chunks, self.padding_value)
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
        
        # Ensure model is on the correct device
        _ensure_model_on_device(self.model, self.device)
        
        # Make predictions
        from pytorch_lightning import Trainer
        trainer = Trainer(accelerator="auto", devices="auto")
        predictions = trainer.predict(self.model, dataloaders=dataloader)
        predictions = torch.cat(predictions).cpu().numpy()
        
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

    def get_feature_importance(self, df: pd.DataFrame = None, mode: str = "gradient") -> dict:
        """
        Get feature importance for the data.
        
        Args:
            df (pandas.DataFrame, optional): DataFrame to analyze. If None, uses training data.
            mode (str): Analysis mode. Options: "gradient" (default) or "attention" (requires exogenous features).
            
        Returns:
            dict: Feature importance scores with keys:
                - 'text_importance': numpy array of shape (n_samples, seq_len)
                - 'exogenous_importance': numpy array of shape (n_samples, n_features) if exogenous features exist
        """
        if self.model is None:
            raise ValueError("Model must be fitted before getting feature importance")
        
        # Use training data if no DataFrame provided
        if df is None:
            if not hasattr(self, '_training_data'):
                raise ValueError("No training data available. Please provide a DataFrame or fit the model first.")
            df = self._training_data
        
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        if 'text' not in df.columns:
            raise ValueError("DataFrame must have a 'text' column")
        
        # Validate mode
        if mode not in ["gradient", "attention"]:
            raise ValueError("Mode must be 'gradient' or 'attention'")
        
        if mode == "attention" and self.exogenous_features is None:
            raise ValueError("Attention mode requires exogenous features")
        
        texts = df['text'].tolist()
        
        # Process texts using the same pipeline as predict()
        encoded_sequences = []
        for text in texts:
            if self.chunk_info:
                max_length, overlap = self.chunk_info
                chunks = chunk_text(text, max_length, overlap)
            else:
                chunks = [text]
            chunks = pad_chunks(chunks, self.padding_value)
            encoded_chunks = [self.encoder.encode(chunk) for chunk in chunks]
            encoded_chunks = [chunk if isinstance(chunk, torch.Tensor) else torch.tensor(chunk)
                            for chunk in encoded_chunks]
            encoded_sequences.append(encoded_chunks)
        
        # Convert to tensor format with proper padding
        sequence_tensors = [torch.stack(seq) for seq in encoded_sequences]
        x = pad_sequence(sequence_tensors, batch_first=True)
        
        # Ensure model and tensors are on the correct device
        _ensure_model_on_device(self.model, self.device)
        x = _ensure_tensor_on_device(x, self.device)
        
        # Process exogenous features
        exogenous = None
        if self.exogenous_features is not None:
            exo_data = df[self.exogenous_features].values
            exo_data_scaled = self.exo_scaler.transform(exo_data)
            exogenous = _ensure_tensor_on_device(torch.tensor(exo_data_scaled, dtype=torch.float32), self.device)
        
        # Get importance based on mode
        if mode == "gradient":
            importance = self.model.get_gradient_importance(x, exogenous)
        elif mode == "attention":
            if not self.model.cross_attention_enabled:
                raise ValueError("Attention mode requires cross-attention to be enabled")
            attention_weights = self.model.get_attention_weights(x, exogenous)
            if attention_weights is None:
                raise ValueError("Could not extract attention weights")
            # Convert attention weights to importance format
            importance = {
                'text_importance': attention_weights.mean(dim=1).squeeze(-1),  # Average over heads
                'exogenous_importance': attention_weights.mean(dim=1).squeeze(-1) if exogenous is not None else None
            }
        
        # Convert to numpy
        result = {}
        for key, tensor in importance.items():
            if tensor is not None:
                result[key] = tensor.detach().cpu().numpy()
        
        return result

    def set_device(self, device: str):
        """
        Change the device for the model and ensure all components are on the new device.
        
        Args:
            device (str): New device specification. Options:
                - "auto": Automatically select best available device
                - "cpu": Force CPU usage
                - "cuda": Force CUDA usage (if available)
                - "cuda:0", "cuda:1", etc.: Specific CUDA device
        """
        new_device = _setup_device(device)
        if new_device != self.device:
            self.device = new_device
            if self.model is not None:
                _ensure_model_on_device(self.model, self.device)
            print(f"Model moved to device: {self.device}")

    def get_device(self) -> torch.device:
        """
        Get the current device of the model.
        
        Returns:
            torch.device: Current device
        """
        return self.device
