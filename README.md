[![PyPI](https://img.shields.io/pypi/v/textregress)](https://pypi.org/project/textregress/)
[![Downloads](https://pepy.tech/badge/textregress)](https://pepy.tech/project/textregress)
[![CI](https://github.com/jinhangjiang/textregress/workflows/CI/badge.svg)](https://github.com/jinhangjiang/textregress/actions)
[![Test](https://github.com/jinhangjiang/textregress/workflows/Test%20Package/badge.svg)](https://github.com/jinhangjiang/textregress/actions)


# TextRegress v1.2.0

TextRegress is a Python package designed to help researchers perform advanced regression analysis on text data. It provides a unified deep learning framework to handle long-text data and supports:
- **Modular Architecture**: Clean, extensible package structure with registry systems for models, encoders, and losses
- **Multiple Model Types**: LSTM and GRU models with full feature parity including cross-attention and feature mixing
- **Configurable text encoding** using SentenceTransformer, TFIDF, or any pretrained Hugging Face models
- **Automatic text chunking** for long documents
- **Deep learning backend** based on PyTorch Lightning with RNN layers
- **Integration of exogenous features** through normalization and attention mechanisms
- **Explainability features** including gradient-based importance and integrated gradients
- **Model persistence** with save/load functionality
- **Sklearn-like API** with `fit`, `predict`, and `fit_predict` methods

## Citation
If this package was helpful in your work, feel free to cite it as

- Jiang, Jinhang and Liu, Ben and Peng, Weiyao and Srinivasan, Karthik, TextRegress: A Python package for advanced regression analysis on long-form text data (May 5, 2025). Software Impacts. https://doi.org/10.1016/j.simpa.2025.100760

```
@article{jiang2025textregress,
  title={TextRegress: A Python package for advanced regression analysis on long-form text data},
  author={Jiang, Jinhang and Liu, Ben and Peng, Weiyao and Srinivasan, Karthik},
  journal={Software Impacts},
  pages={100760},
  year={2025},
  publisher={Elsevier}
}
```

## Installation

TextRegress requires Python 3.6 or higher. You can install it directly from PyPI:

```bash
pip install textregress
```

Or install from the repository:

```bash
git clone https://github.com/jinhangjiang/textregress.git
cd textregress
pip install -e .
```

## Quick Start

```python
import pandas as pd
from textregress import TextRegressor

# Create sample data
data = {
    'text': [
        "This is a positive review about the product.",
        "The quality is excellent and I recommend it.",
        "Not satisfied with the purchase.",
        "Great value for money."
    ],
    'y': [4.5, 4.8, 2.1, 4.2],
    'feature1': [1.0, 1.2, 0.8, 1.1],
    'feature2': [0.5, 0.6, 0.3, 0.7]
}
df = pd.DataFrame(data)

# Create and train the model
regressor = TextRegressor(
    model_name="lstm",  # or "gru"
    encoder_model="sentence-transformers/all-MiniLM-L6-v2",
    exogenous_features=["feature1", "feature2"],
    max_steps=100
)

# Fit and predict
predictions = regressor.fit_predict(df)
print(f"Predictions: {predictions}")
```

## Implementation

TextRegress Model (encoder_model, encoder_params=None, rnn_type, rnn_layers, hidden_size, bidirectional,  
    inference_layer_units, exogenous_features=None, feature_mixer=False, learning_rate: float, 
    loss_function: Union[str, Callable], encoder_output_dim: int, optimizer_name: str, optimizer_params: dict=None,  
    cross_attention_enabled: bool=False, cross_attention_layer: Optional[nn.Module]=None, dropout_rate: float=0.0,  
    se_layer: bool=True, random_seed: int=1)

**Parameters:**

- **encoder_model**: *str*  
  Specifies the pretrained encoder model to use. This can be a HuggingFace model identifier (e.g., `"sentence-transformers/all-MiniLM-L6-v2"`) or `"tfidf"` for a TFIDF-based encoder.

- **encoder_params**: *Optional[dict]*  
  A dictionary of additional parameters for configuring the encoder. For example, when using a TFIDF encoder, users can supply parameters such as `{"max_features": 1000, "ngram_range": (1, 2)}`. These parameters are passed directly to the underlying encoder.

- **rnn_type**: *str*  
  Specifies the type of recurrent unit to use. Acceptable values include `"LSTM"` and `"GRU"`. This choice determines the basic building block of the temporal processing module.

- **rnn_layers**: *int*  
  The number of stacked RNN layers in the model. More layers can capture higher-order temporal features but may require more data and computation.

- **hidden_size**: *int*  
  The number of hidden units in each RNN layer. This parameter defines the dimensionality of the hidden state and directly influences the model’s capacity.

- **bidirectional**: *bool*  
  When set to `True`, the RNN operates in a bidirectional manner, processing the sequence in both forward and backward directions. This effectively doubles the output dimension of the RNN.

- **inference_layer_units**: *int*  
  The number of units in the intermediate inference (fully connected) layer. This layer transforms the processed features into a representation suitable for the final regression output.

- **exogenous_features**: *Optional[List[str]]*  
  A list of column names representing additional (exogenous) features to be incorporated into the model.  
  - When `cross_attention_enabled` is `True`, these features are projected to match the RNN output dimension and integrated via a cross-attention mechanism.  
  - When `cross_attention_enabled` is `False` and `feature_mixer` is also `False`, the normalized exogenous features are directly concatenated with the document embedding.  
  - When `feature_mixer` is `True`, the model first computes an inference output from the document embedding and then mixes in the normalized exogenous features via an additional mixing layer before making predictions.

- **feature_mixer**: *bool*  
  A flag to enable additional mixing of exogenous features. When set to `True`, the model mixes normalized exogenous features with the inference output of the document embedding via a dedicated linear layer. When `False`, the exogenous features are concatenated directly with the document embedding.

- **learning_rate**: *float*  
  The learning rate used by the optimizer during training. This controls how quickly the model weights are updated.

- **loss_function**: *Union[str, Callable]*  
  Specifies the loss function for training. Supported string options include `"mae"`, `"mse"`, `"rmse"`, `"smape"`, `"wmape"`, and `"mape"`. Alternatively, users can provide a custom loss function as a callable. Custom loss functions must accept keyword arguments `pred` and `target`.

- **encoder_output_dim**: *int*  
  The dimensionality of the vector output from the encoder module. This value is used to configure the input size of the RNN. For instance, when using a TFIDF encoder, this is automatically set based on the size of the fitted vocabulary.

- **optimizer_name**: *str*  
  The name of the optimizer to be used (e.g., `"adam"`, `"sgd"`, etc.). The model dynamically searches within PyTorch’s optimizers to instantiate the specified optimizer.

- **optimizer_params**: *dict*  
  A dictionary containing additional keyword parameters to pass to the optimizer upon instantiation (for example, momentum for SGD).

- **cross_attention_enabled**: *bool*  
  A flag indicating whether to enable a cross-attention mechanism. When `True`, the model generates a global token (by averaging the RNN outputs) and uses it as the query to attend over the projected exogenous features. The output of this attention is concatenated with the RNN’s last time-step output before further processing.

- **cross_attention_layer**: *Optional[nn.Module]*  
  An optional custom cross-attention layer. If not provided and cross attention is enabled, a default single-head MultiheadAttention layer (from `nn.MultiheadAttention`) is used.

- **dropout_rate**: *float*  
  The dropout rate applied after each major component (e.g., after the RNN output, global token generation, inference layers, cross-attention, and squeeze-and-excitation block). A value of 0.0 means no dropout is applied.

- **se_layer**: *bool*  
  Specifies whether to enable the squeeze-and-excitation (SE) block on the output of the inference layer. When enabled, the SE block recalibrates channel-wise feature responses, potentially enhancing model performance.

- **random_seed**: *int*  
  Sets the random seed for reproducibility. This value is used to initialize PyTorch (via `torch.manual_seed`), ensuring that training results are consistent across runs.

---

**Usage Example:**

```python
from textregress import TextRegressor

# Instantiate the TextRegressor with custom encoder parameters and feature mixing:
regressor = TextRegressor(
    encoder_model="tfidf",  # Use the TFIDF encoder
    encoder_params={"max_features": 1000, "ngram_range": (1, 2)},  # Custom TFIDF parameters
    rnn_type="GRU",                     # Use GRU instead of LSTM
    rnn_layers=2,                       # Use 2 RNN layers
    hidden_size=100,                    # Hidden size set to 100
    bidirectional=False,                # Unidirectional RNN
    inference_layer_units=50,           # Inference layer with 50 units
    chunk_info=(100, 25),               # Chunk text into segments of 100 words with an overlap of 25 words
    padding_value=0,                    # Padding value for chunks
    exogenous_features=["ex1", "ex2"],  # Include two exogenous features
    feature_mixer=True,                 # Enable feature mixer to combine document embedding with exogenous features
    learning_rate=0.001,                # Learning rate of 0.001
    loss_function="mae",                # MAE loss (or a custom callable loss function)
    encoder_output_dim=1000,            # For TFIDF, this is set to the number of features (e.g., 1000)
    optimizer_name="adam",              # Use Adam optimizer
    cross_attention_enabled=True,       # Enable cross attention between a global token and exogenous features
    cross_attention_layer=None,         # Use default cross attention layer
    dropout_rate=0.1,                   # Apply dropout with a rate of 0.1
    se_layer=True,                      # Enable the squeeze-and-excitation block
    random_seed=42                      # Set a random seed for reproducibility
)

# Fit the model on a DataFrame.
regressor.fit(df, val_size=0.2)

# Predict on the same DataFrame.
predictions = regressor.predict(df)
```

## New in v1.2.0

- **🔄 Modular Architecture**: Complete restructuring into modular packages (`models/`, `encoders/`, `losses/`, `utils/`) with registry systems
- **🧠 GRU Model**: New GRU implementation with full parity to LSTM including cross-attention and feature mixing
- **🔍 Explainability**: Lightweight explainability features including gradient-based importance and integrated gradients
- **💾 Model Persistence**: Save and load models with full PyTorch parameter exposure
- **📊 Embedding Extraction**: Extract document and sequence embeddings for transfer learning
- **⚡ Enhanced API**: Improved registry system and cleaner import structure

## Features

- **Unified DataFrame Interface**  
  The estimator methods (`fit`, `predict`, `fit_predict`) accept a single pandas DataFrame with:
  - **`text`**: Input text data (can be long-form text).
  - **`y`**: Continuous target variable.
  - Additional columns can be provided as exogenous features.

- **Configurable Text Encoding**  
  Choose from multiple encoding methods:
  - **TFIDF Encoder:** Activated when the model identifier contains `"tfidf"`.
  - **SentenceTransformer Encoder:** Activated when the model identifier contains `"sentence-transformers"`.
  - **Generic Hugging Face Encoder:** Supports any pre-trained Hugging Face model using `AutoTokenizer`/`AutoModel` with a mean-pooling strategy.

- **Text Chunking**  
  Automatically splits long texts into overlapping, fixed-size chunks (only full chunks are processed) to ensure consistent input size.

- **Deep Learning Regression Model**  
  Utilizes an RNN-based (LSTM/GRU) network implemented with PyTorch Lightning:
  - Customizable number of layers, hidden size, and bidirectionality.
  - Optionally integrates exogenous features into the regression process.

- **Custom Loss Functions**  
  Multiple loss functions are available via `loss.py`:
  - MAE (default)
  - SMAPE
  - MSE
  - RMSE
  - wMAPE
  - MAPE

- **Training Customization**  
  Fine-tune training behavior with parameters such as:
  - `max_steps`: Maximum training steps (default: 500).
  - `early_stop_enabled`: Enable early stopping (default: False).
  - `patience_steps`: Steps with no improvement before stopping (default: 10 when early stopping is enabled).
  - `val_check_steps`: Validation check interval (default: 50, automatically adjusted if needed).
  - `val_size`: Proportion of data reserved for validation when early stopping is enabled.

- **GPU Auto-Detection**  
  Automatically leverages available GPUs via PyTorch Lightning (using `accelerator="auto"` and `devices="auto"`).


