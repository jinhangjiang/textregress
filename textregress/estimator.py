import math
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import random
import numpy as np

from .encoding import get_encoder
from .models import TextRegressionModel
from .utils import chunk_text, pad_chunks, TextRegressionDataset, collate_fn

class TextRegressor:
    """
    A text regression estimator following an sklearn-like API.
    
    This estimator takes in a pandas DataFrame containing a 'text' column and a 'y'
    column (with optional exogenous feature columns) and processes the text using configurable
    encoding and chunking, then applies a deep learning model (with RNN-based layers) to predict
    the target variable.
    
    Additional encoder parameters can be passed via `encoder_params`. Also, the loss_function
    parameter can be provided as either a string (one of "mae", "mse", "rmse", "smape", "mape", "wmape")
    or as a custom callable loss function.
    """
    def __init__(self, 
                 encoder_model="sentence-transformers/all-MiniLM-L6-v2",
                 encoder_params=None,
                 rnn_type="LSTM",
                 rnn_layers=2,
                 hidden_size=512,
                 bidirectional=True,
                 inference_layer_units=100,
                 chunk_info=None,
                 padding_value=0,
                 exogenous_features=None,
                 learning_rate=1e-3,
                 loss_function="mae",  # can be a string or a callable
                 max_steps=500,
                 early_stop_enabled=False,
                 patience_steps=None,
                 val_check_steps=50,
                 optimizer_name="adam",
                 optimizer_params={},
                 cross_attention_enabled=False,
                 cross_attention_layer=None,
                 dropout_rate=0.0,
                 se_layer=True,
                 random_seed=1,
                 **kwargs):
        """
        Initialize the TextRegressor.
        
        Args:
            encoder_model (str): Pretrained encoder model identifier.
            encoder_params (dict, optional): Additional parameters to configure the encoder.
            rnn_type (str): Type of RNN to use ("LSTM" or "GRU").
            rnn_layers (int): Number of RNN layers.
            hidden_size (int): Hidden size for the RNN.
            bidirectional (bool): Whether to use bidirectional RNN.
            inference_layer_units (int): Units in the final linear inference layer.
            chunk_info (tuple, optional): (chunk_size, overlap) for splitting long texts.
            padding_value (int, optional): Padding value for text chunks.
            exogenous_features (list, optional): List of additional exogenous feature column names.
            learning_rate (float): Learning rate for the optimizer.
            loss_function (str or callable): Loss function to use. If a string, one of "mae", "mse", "rmse",
                "smape", "mape", "wmape" is supported; if a callable is provided, it will be used directly.
            max_steps (int): Maximum number of training steps. Default is 500.
            early_stop_enabled (bool): Whether to enable early stopping. Default is False.
            patience_steps (int, optional): Number of steps with no improvement before stopping (default: 10 if enabled).
            val_check_steps (int): Interval for validation checks.
            optimizer_name (str): Name of the optimizer to use (e.g., "adam", "sgd").
            optimizer_params (dict): Additional keyword arguments for the optimizer.
            cross_attention_enabled (bool): Whether to enable cross attention between a global token and exogenous features.
            cross_attention_layer (nn.Module, optional): Custom cross attention layer.
            dropout_rate (float): Dropout rate to apply after each component.
            se_layer (bool): Whether to enable the squeeze-and-excitation block.
            random_seed (int): Random seed for reproducibility.
            **kwargs: Additional keyword arguments.
        """
        # Set random seed for reproducibility.
        self.random_seed = random_seed
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        
        self.encoder_model = encoder_model
        self.encoder_params = encoder_params if encoder_params is not None else {}
        self.rnn_type = rnn_type
        self.rnn_layers = rnn_layers
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.inference_layer_units = inference_layer_units
        self.chunk_info = chunk_info
        self.padding_value = padding_value
        self.exogenous_features = exogenous_features
        self.learning_rate = learning_rate
        self.loss_function = loss_function
        self.max_steps = max_steps
        self.early_stop_enabled = early_stop_enabled
        self.val_check_steps = val_check_steps
        self.optimizer_name = optimizer_name
        self.optimizer_params = optimizer_params
        self.cross_attention_enabled = cross_attention_enabled
        self.cross_attention_layer = cross_attention_layer
        self.dropout_rate = dropout_rate
        self.se_layer = se_layer
        
        if self.early_stop_enabled:
            self.patience_steps = patience_steps if patience_steps is not None else 10
        else:
            self.patience_steps = None
        
        # Instantiate the encoder with custom parameters.
        self.encoder = get_encoder(self.encoder_model, **self.encoder_params)
        self.model = None
        self.exo_scaler = None

    def fit(self, df, batch_size=64, val_size=None, **kwargs):
        """
        Fit the TextRegressor model on the provided DataFrame.
        
        The DataFrame must have a 'text' column and a 'y' column.
        Optionally, it can include additional exogenous feature columns.
        
        Args:
            df (pandas.DataFrame): DataFrame containing 'text' and 'y' columns.
            batch_size (int): Batch size for training. Default is 64.
            val_size (float, optional): Proportion (between 0 and 1) of data to use for validation.
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
        
        # Fit the encoder if necessary (e.g., for TFIDF).
        if hasattr(self.encoder, 'fitted') and not self.encoder.fitted:
            corpus = []
            for text in texts:
                chunks = chunk_text(text, self.chunk_info, encoder=self.encoder)
                chunks = pad_chunks(chunks, padding_value=self.padding_value)
                corpus.extend(chunks)
            self.encoder.fit(corpus)
        
        encoded_sequences = []
        for text in tqdm(texts, desc="Processing texts"):
            chunks = chunk_text(text, self.chunk_info, encoder=self.encoder)
            chunks = pad_chunks(chunks, padding_value=self.padding_value)
            encoded_chunks = [self.encoder.encode(chunk) for chunk in chunks]
            encoded_chunks = [chunk if isinstance(chunk, torch.Tensor) else torch.tensor(chunk)
                              for chunk in encoded_chunks]
            encoded_sequences.append(encoded_chunks)
        
        if self.exogenous_features is not None:
            exo_data = df[self.exogenous_features].values
            self.exo_scaler = StandardScaler()
            exo_data_scaled = self.exo_scaler.fit_transform(exo_data)
            exo_list = [list(row) for row in exo_data_scaled]
        else:
            exo_list = None
        
        dataset = TextRegressionDataset(encoded_sequences, targets, exogenous=exo_list)
        
        if self.early_stop_enabled:
            if val_size is None:
                raise ValueError("When early_stop_enabled is True, you must specify val_size (a float between 0 and 1).")
            indices = list(range(len(dataset)))
            train_idx, val_idx = train_test_split(indices, test_size=val_size, random_state=self.random_seed)
            train_subset = Subset(dataset, train_idx)
            val_subset = Subset(dataset, val_idx)
            train_loader = DataLoader(train_subset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)
        else:
            train_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
            val_loader = None
        
        steps_per_epoch = len(train_loader)
        computed_epochs = math.ceil(self.max_steps / steps_per_epoch)
        
        if hasattr(self.encoder, 'model') and hasattr(self.encoder.model, 'get_sentence_embedding_dimension'):
            encoder_output_dim = self.encoder.model.get_sentence_embedding_dimension()
        elif hasattr(self.encoder, 'output_dim'):
            encoder_output_dim = self.encoder.output_dim
        else:
            encoder_output_dim = 768
        
        self.model = TextRegressionModel(
            rnn_type=self.rnn_type,
            rnn_layers=self.rnn_layers,
            hidden_size=self.hidden_size,
            bidirectional=self.bidirectional,
            inference_layer_units=self.inference_layer_units,
            exogenous_features=self.exogenous_features,
            learning_rate=self.learning_rate,
            loss_function=self.loss_function,
            encoder_output_dim=encoder_output_dim,
            optimizer_name=self.optimizer_name,
            optimizer_params=self.optimizer_params,
            cross_attention_enabled=self.cross_attention_enabled,
            cross_attention_layer=self.cross_attention_layer,
            dropout_rate=self.dropout_rate,
            se_layer=self.se_layer,
            random_seed=self.random_seed
        )
        
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
        
        if self.early_stop_enabled:
            val_check_interval = min(self.val_check_steps, len(train_loader))
        else:
            val_check_interval = None
        
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

    def predict(self, df, batch_size=64, **kwargs):
        """
        Predict continuous target values for new text data in the provided DataFrame.
        
        The DataFrame must have a 'text' column (and optionally exogenous feature columns).
        
        Args:
            df (pandas.DataFrame): DataFrame containing a 'text' column.
            batch_size (int): Batch size for prediction. Default is 64.
            **kwargs: Additional arguments for prediction.
            
        Returns:
            predictions (list): Predicted continuous values.
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        if 'text' not in df.columns:
            raise ValueError("DataFrame must have a 'text' column")
        
        texts = df['text'].tolist()
        encoded_sequences = []
        for text in tqdm(texts, desc="Processing texts"):
            chunks = chunk_text(text, self.chunk_info, encoder=self.encoder)
            chunks = pad_chunks(chunks, padding_value=self.padding_value)
            encoded_chunks = [self.encoder.encode(chunk) for chunk in chunks]
            encoded_chunks = [chunk if isinstance(chunk, torch.Tensor) else torch.tensor(chunk)
                              for chunk in encoded_chunks]
            encoded_sequences.append(encoded_chunks)
        
        if self.exogenous_features is not None:
            exo_data = df[self.exogenous_features].values
            exo_data_scaled = self.exo_scaler.transform(exo_data)
            exo_list = [list(row) for row in exo_data_scaled]
        else:
            exo_list = None
        
        dataset = TextRegressionDataset(encoded_sequences, [0] * len(encoded_sequences), exogenous=exo_list)
        predict_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)
        
        from pytorch_lightning import Trainer
        trainer = Trainer(accelerator="auto", devices="auto")
        predictions = trainer.predict(self.model, dataloaders=predict_loader)
        flat_predictions = [pred.item() for batch in predictions for pred in batch]
        return flat_predictions

    def fit_predict(self, df, **kwargs):
        """
        Fit the model on the provided DataFrame and immediately predict on it.
        
        Args:
            df (pandas.DataFrame): DataFrame containing 'text' and 'y' columns.
            **kwargs: Additional arguments.
            
        Returns:
            predictions (list): Predicted continuous values.
        """
        self.fit(df, **kwargs)
        return self.predict(df, **kwargs)
