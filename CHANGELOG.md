# Changelog

All notable changes to this project will be documented in this file.

## [v1.1.1] - 2025-02-22

### Added
- **Custom TFIDF Parameters:**  
  Users can now pass custom parameters via `encoder_params` to configure the TFIDF encoder (e.g., `max_features`, `ngram_range`).
- **Custom Loss Function Support:**  
  The estimator now accepts a custom loss function (as a callable). This loss function is automatically invoked with keyword arguments `pred` and `target`.
- **Feature Mixer for Exogenous Features:**  
  A new parameter `feature_mixer` allows mixing normalized exogenous features with the document embedding:
  - When `feature_mixer = False`, normalized exogenous features are concatenated directly with the document embedding.
  - When `feature_mixer = True`, the inference output from the document embedding is mixed with normalized exogenous features via an additional mixing layer before making predictions.

### Changed
- **TFIDF Encoder Enhancements:**  
  - Automatically fits on the entire corpus if not already fitted.
  - Converts output to a `torch.Tensor` with `float32` dtype.
  - Dynamically sets its `output_dim` based on the fitted vocabulary size.
- **Dynamic Encoder Output Dimension:**  
  The model now uses the actual encoder output dimension (instead of a hardcoded 768) to configure the RNN.
- **Exogenous Feature Handling:**  
  Improved integration of exogenous features with two distinct pathways depending on `cross_attention_enabled` and `feature_mixer` settings.
- **Loss Function Invocation:**  
  The training and validation steps now pass the prediction and target values as keyword arguments (`pred` and `target`) to the loss function.

### Fixed
- **Runtime Dimension Mismatch:**  
  Resolved errors such as "Expected 768, got X" by dynamically determining and using the proper encoder output dimension.
- **Loss Function Integration:**  
  Fixed issues where custom loss functions did not automatically pick up the correct inputs.


## [Version 1.1.0] - 2025-02-19

### Added
- **Dynamic Optimizer Support:**  
  The `configure_optimizers` method now dynamically searches through `torch.optim` to support any optimizer available in PyTorch. Users can now specify the optimizer by name (e.g., "adam", "sgd", etc.) along with custom optimizer parameters without modifying the source code.
  
- **Random Seed Parameter:**  
  A new `random_seed` parameter (default: 1) has been added to both the estimator and model. This parameter sets the seed for Python’s `random` module, NumPy, and PyTorch to ensure reproducibility across runs.

- **Dropout Layers:**  
  Dropout layers have been integrated after every major component of the model (after the RNN output, global token generation, inference layers, and SE block) to improve generalization. Users can control the dropout rate using the `dropout_rate` parameter (default: 0.0).

- **Cross-Attention Mechanism:**  
  An optional cross-attention mechanism has been added to enhance the integration of exogenous features. When enabled via `cross_attention_enabled`, the model:
  - Generates a global token by averaging RNN outputs.
  - Projects exogenous features and applies cross attention between the global token and these features.
  - Concatenates the cross-attention output with the RNN’s final output before the inference layer.
  
  Users can also provide a custom cross-attention layer using `cross_attention_layer`.

- **Squeeze-and-Excitation (SE) Block:**  
  An optional SE block has been incorporated to further recalibrate channel-wise features and boost model performance. This can be enabled or disabled via the `se_layer` parameter (default: True).

### Changed
- **Model Architecture Enhancements:**  
  The overall architecture of the model has been updated to integrate dropout, cross-attention, and an SE block. This makes the model more robust and flexible for various regression tasks on text data.
  
- **Enhanced Parameterization and Documentation:**  
  Both the estimator and model now expose additional parameters (such as optimizer customization, random seed, dropout rate, cross-attention options, and SE block control), with improved inline documentation to facilitate ease of use and future maintenance.

### Summary
Version 1.1.0 of TextRegress introduces significant enhancements that empower users with greater flexibility and reproducibility. With dynamic optimizer support, a reproducibility mechanism via the random seed parameter, and architectural improvements including dropout, cross-attention, and a squeeze-and-excitation block, this update positions TextRegress as a more robust and adaptable tool for regression analysis on text data.


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