# Changelog

All notable changes to this project will be documented in this file.

## [Version 1.0.0] - 2025-02-15

### Added
- **Unified DataFrame Interface:**  
  The estimator now accepts a single pandas DataFrame with the following required columns:
  - `text`: Input text data.
  - `y`: Continuous target variable.
  
  Additional columns can be provided as exogenous features.

- **Enhanced Training Customization:**  
  New parameters have been added to control training behavior:
  - `max_steps`: Maximum number of training steps (default: 500).
  - `early_stop_enabled`: Option to enable early stopping (default: False).
  - `patience_steps`: Number of steps with no improvement before stopping (default: 10 when early stopping is enabled).
  - `val_check_steps`: Interval for validation checks (default: 50, automatically adjusted if necessary).
  - `val_size`: Percentage split for validation when early stopping is enabled.

- **Loss Module (`loss.py`):**  
  Multiple loss functions are now available:
  - MAE (default)
  - SMAPE
  - MSE
  - RMSE
  - wMAPE
  - MAPE

- **Generalized Encoder Support:**  
  The encoder factory (`get_encoder`) now supports three cases:
  - **TFIDF Encoder:** Activated when the model identifier contains `"tfidf"`.
  - **SentenceTransformer Encoder:** Activated when the model identifier contains `"sentence-transformers"`.
  - **Generic Hugging Face Encoder:** Supports any pre-trained Hugging Face model using `AutoTokenizer` and `AutoModel` with a generic mean-pooling strategy.

- **GPU Auto-Detection:**  
  The training configuration automatically leverages GPUs if available (using `accelerator="auto"` and `devices="auto"`).

- **Dynamic Embedding Dimension Handling:**  
  The model dynamically detects the encoder’s output dimension (e.g., 384 for `"sentence-transformers/all-MiniLM-L6-v2"`) and configures the RNN input size accordingly.

- **Improved Chunking Functionality:**  
  The `chunk_text` function has been updated to yield only full chunks, ensuring consistent input sizes.

- **Extensive Testing Suite:**  
  Comprehensive tests have been provided for utility functions, encoder types, and the full estimator functionality. These tests ensure the package works as expected and are designed to be easily maintained and extended.

### Changed
- **Default Batch Size:**  
  The default batch size has been updated to 64.
  
- **Interface Simplification:**  
  The estimator’s interface has been updated to use a single DataFrame input instead of separate `X` and `y` parameters.

---

This initial release (Version 1.0.0) is designed with modularity and maintainability in mind, making it easy to extend and update in the future.