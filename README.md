[![PyPI](https://img.shields.io/pypi/v/textregress)](https://pypi.org/project/textregress/)
[![Downloads](https://pepy.tech/badge/textregress)](https://pepy.tech/project/textregress)
# TextRegress

TextRegress is a Python package designed to help researchers perform linear regression analysis on text data. It supports:
- Configurable text encoding using SentenceTransformer or custom methods (e.g., TFIDF).
- Automatic text chunking for long documents.
- A deep learning backend based on PyTorch Lightning with RNN (LSTM/GRU) layers.
- Integration of exogenous features through standard normalization and attention mechanisms.
- An sklearn-like API with `fit`, `predict`, and `fit_predict` methods.

## Installation

TextRegress requires Python 3.6 or higher. You can install it directly from the repository:

```bash
git clone https://github.com/yourusername/TextRegress.git
cd TextRegress
pip install -e .
```

You may also install it through pypi:

```python
pip install textregress
```

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

- **Dynamic Embedding Dimension Handling**  
  The model dynamically detects the encoder’s output dimension (e.g., 384 for `"sentence-transformers/all-MiniLM-L6-v2"`) and configures the RNN input accordingly.

- **Extensive Testing Suite**  
  Comprehensive tests ensure that utility functions, encoder types, and estimator functionality work as expected, making it easy to maintain and extend the package.


