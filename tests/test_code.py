import pandas as pd
import torch
import numpy as np

# --- Import functions/classes from the package ---
from textregress.encoders import get_encoder, list_available_encoders
from textregress.models import get_model, list_available_models
from textregress.losses import get_loss_function, list_available_losses
from textregress.utils import chunk_text, pad_chunks, collate_fn, TextRegressionDataset
from textregress.estimator import TextRegressor

# ---------------------------------
# Utility Function Tests (utils.py)
# ---------------------------------
def test_chunk_text():
    text = "one two three four five six seven eight nine"
    # Test word-based chunking (original implementation)
    chunks = chunk_text(text, max_length=3, overlap=1)
    expected = ["one two three", "three four five", "five six seven", "seven eight nine"]
    assert chunks == expected, f"Expected {expected}, got {chunks}"
    
    # Test with no overlap
    chunks_no_overlap = chunk_text(text, max_length=3, overlap=0)
    expected_no_overlap = ["one two three", "four five six", "seven eight nine"]
    assert chunks_no_overlap == expected_no_overlap, f"Expected {expected_no_overlap}, got {chunks_no_overlap}"

def test_pad_chunks():
    chunks = ["a", "b", "c"]
    padded = pad_chunks(chunks, padding_value=0)
    assert padded == chunks, "pad_chunks should return input as-is"

def test_collate_fn():
    # Create a dummy batch of two samples. Each sample is a tuple: (encoded_sequence, target).
    sample1 = ([torch.ones(10), torch.ones(10) * 2], 1.0)
    sample2 = ([torch.ones(10) * 3, torch.ones(10) * 4], 2.0)
    batch = [sample1, sample2]
    
    padded_sequences, targets = collate_fn(batch)
    # Expected shape: (batch_size, max_seq_len, feature_dim) = (2, 2, 10)
    assert padded_sequences.shape == (2, 2, 10), f"Got shape {padded_sequences.shape}"
    assert targets.shape[0] == 2

def test_text_regression_dataset():
    # Create a dummy dataset with two samples, each with two encoded chunks.
    sample1 = [torch.zeros(5), torch.ones(5)]
    sample2 = [torch.ones(5) * 2, torch.ones(5) * 3]
    targets = [10.0, 20.0]
    dataset = TextRegressionDataset([sample1, sample2], targets)
    assert len(dataset) == 2, "Dataset should have two samples"
    sample, target = dataset[0]
    assert isinstance(sample, list), "Each sample should be a list"
    assert sample[0].shape[0] == 5, "Chunk tensor should have shape (5,)"
    assert target == 10.0

# ---------------------------------
# Registry Tests
# ---------------------------------
def test_encoder_registry():
    """Test encoder registry functionality"""
    available_encoders = list_available_encoders()
    assert 'sentence_transformer' in available_encoders, "sentence_transformer should be registered"
    assert 'tfidf' in available_encoders, "tfidf should be registered"
    
    # Test getting encoders
    sentence_encoder = get_encoder("sentence_transformer", model_name="sentence-transformers/all-MiniLM-L6-v2")
    assert sentence_encoder is not None, "Should be able to get sentence transformer encoder"
    
    tfidf_encoder = get_encoder("tfidf")
    assert tfidf_encoder is not None, "Should be able to get TF-IDF encoder"

def test_model_registry():
    """Test model registry functionality"""
    available_models = list_available_models()
    assert 'lstm' in available_models, "lstm should be registered"
    assert 'gru' in available_models, "gru should be registered"
    
    # Test getting models
    lstm_model = get_model("lstm")
    assert lstm_model is not None, "Should be able to get LSTM model"
    
    gru_model = get_model("gru")
    assert gru_model is not None, "Should be able to get GRU model"

def test_loss_registry():
    """Test loss registry functionality"""
    available_losses = list_available_losses()
    assert 'mae' in available_losses, "mae should be registered"
    assert 'mse' in available_losses, "mse should be registered"
    assert 'rmse' in available_losses, "rmse should be registered"
    
    # Test getting losses
    mae_loss = get_loss_function("mae")
    assert mae_loss is not None, "Should be able to get MAE loss"
    
    mse_loss = get_loss_function("mse")
    assert mse_loss is not None, "Should be able to get MSE loss"

# ---------------------------------
# Encoder Tests
# ---------------------------------
def test_tfidf_encoder():
    """Test TF-IDF encoder functionality"""
    encoder = get_encoder("tfidf")
    texts = ["hello world", "test sentence", "another test"]
    encoder.fit(texts)
    vector = encoder.encode("hello world")
    # Expect a tensor
    assert torch.is_tensor(vector), "TF-IDF encoder should return a tensor"
    assert vector.ndim == 1, "TF-IDF vector should be 1D"
    assert vector.shape[0] > 0, "TF-IDF vector should have positive dimension"

def test_sentence_transformer_encoder():
    """Test sentence transformer encoder functionality"""
    # This test loads the actual model; may take a few seconds.
    encoder = get_encoder("sentence_transformer", model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector = encoder.encode("Hello world")
    assert torch.is_tensor(vector), "Sentence transformer should return a tensor"
    # Expected output dimension for all-MiniLM-L6-v2 is 384.
    assert vector.shape[-1] == 384, f"Expected 384, got {vector.shape[-1]}"

# ---------------------------------
# Model Tests
# ---------------------------------
def test_lstm_model():
    """Test LSTM model creation and basic functionality"""
    model = get_model("lstm")
    # Test that model can be instantiated with basic parameters
    lstm_instance = model(
        encoder_output_dim=384,
        learning_rate=1e-3,
        loss_function="mae"
    )
    assert lstm_instance is not None, "LSTM model should be instantiable"

def test_gru_model():
    """Test GRU model creation and basic functionality"""
    model = get_model("gru")
    # Test that model can be instantiated with basic parameters
    gru_instance = model(
        encoder_output_dim=384,
        learning_rate=1e-3,
        loss_function="mae"
    )
    assert gru_instance is not None, "GRU model should be instantiable"

# ---------------------------------
# Estimator Integration Tests (estimator.py)
# ---------------------------------
# Create a dummy encoder for fast testing.
class DummyEncoder:
    def __init__(self, **kwargs):
        self.fitted = False
        self.output_dim = 768
    
    def encode(self, text):
        # Return a fixed tensor of ones with dimension 768.
        return torch.ones(768)
    
    def fit(self, texts):
        self.fitted = True

def test_estimator_fit_predict_without_exogenous():
    """Test estimator without exogenous features"""
    df = pd.DataFrame({
        "text": ["Test sentence one.", "Test sentence two."],
        "y": [1.0, 2.0]
    })
    
    # Monkey-patch get_encoder to return DummyEncoder.
    from textregress.encoders import registry
    original_get_encoder = registry.get_encoder
    
    def mock_get_encoder(name, **kwargs):
        return DummyEncoder(**kwargs)
    
    registry.get_encoder = mock_get_encoder
    
    try:
    estimator = TextRegressor(
        encoder_model="sentence-transformers/all-MiniLM-L6-v2",
        max_steps=2,
        early_stop_enabled=False,
        loss_function="mae"
    )
    estimator.fit(df)
    preds = estimator.predict(df)
        assert isinstance(preds, np.ndarray), "Predictions should be numpy array"
        assert len(preds) == len(df), "Number of predictions should match number of samples"
    for pred in preds:
            assert isinstance(pred, (float, np.floating)), "Each prediction should be a float"
    finally:
    # Restore original function.
        registry.get_encoder = original_get_encoder

def test_estimator_fit_predict_with_exogenous_and_early_stop():
    """Test estimator with exogenous features and early stopping"""
    df = pd.DataFrame({
        "text": ["Exogenous test one.", "Exogenous test two.", "Exogenous test three."],
        "y": [1.0, 2.0, 3.0],
        "exo1": [0.1, 0.2, 0.3],
        "exo2": [10, 20, 30]
    })
    
    from textregress.encoders import registry
    original_get_encoder = registry.get_encoder
    
    def mock_get_encoder(name, **kwargs):
        return DummyEncoder(**kwargs)
    
    registry.get_encoder = mock_get_encoder
    
    try:
    estimator = TextRegressor(
        encoder_model="sentence-transformers/all-MiniLM-L6-v2",
        exogenous_features=["exo1", "exo2"],
        max_steps=3,
        early_stop_enabled=True,
        val_check_steps=1,
        loss_function="mse"
    )
    estimator.fit(df, val_size=0.33)
    preds = estimator.predict(df)
        assert isinstance(preds, np.ndarray), "Predictions should be numpy array"
        assert len(preds) == len(df), "Number of predictions should match number of samples"
    for pred in preds:
            assert isinstance(pred, (float, np.floating)), "Each prediction should be a float"
    finally:
        registry.get_encoder = original_get_encoder

def test_estimator_with_tfidf():
    """Test estimator with TF-IDF encoder"""
    df = pd.DataFrame({
        "text": ["TF-IDF test one.", "TF-IDF test two."],
        "y": [1.0, 2.0]
    })
    
    estimator = TextRegressor(
        encoder_model="tfidf",
        max_steps=2,
        early_stop_enabled=False,
        loss_function="mae"
    )
    estimator.fit(df)
    preds = estimator.predict(df)
    assert isinstance(preds, np.ndarray), "Predictions should be numpy array"
    assert len(preds) == len(df), "Number of predictions should match number of samples"

def test_estimator_with_gru_model():
    """Test estimator with GRU model"""
    df = pd.DataFrame({
        "text": ["GRU test one.", "GRU test two."],
        "y": [1.0, 2.0]
    })
    
    from textregress.encoders import registry
    original_get_encoder = registry.get_encoder
    
    def mock_get_encoder(name, **kwargs):
        return DummyEncoder(**kwargs)
    
    registry.get_encoder = mock_get_encoder
    
    try:
        estimator = TextRegressor(
            model_name="gru",
            encoder_model="sentence-transformers/all-MiniLM-L6-v2",
            max_steps=2,
            early_stop_enabled=False,
            loss_function="mae"
        )
        estimator.fit(df)
        preds = estimator.predict(df)
        assert isinstance(preds, np.ndarray), "Predictions should be numpy array"
        assert len(preds) == len(df), "Number of predictions should match number of samples"
    finally:
        registry.get_encoder = original_get_encoder

# ---------------------------------
# Version Test
# ---------------------------------
def test_version():
    """Test that version is accessible"""
    import textregress
    assert hasattr(textregress, '__version__'), "Package should have __version__ attribute"
    assert textregress.__version__ == "1.2.3", f"Expected version 1.2.3, got {textregress.__version__}"

# --------------------------------
# Run Tests if This Script is Main
# --------------------------------
if __name__ == "__main__":
    print("Running tests for textregress 1.2.3...")
    
    # Version test
    test_version()
    print("✓ test_version passed.")
    
    # Utility tests
    test_chunk_text()
    print("✓ test_chunk_text passed.")
    
    test_pad_chunks()
    print("✓ test_pad_chunks passed.")
    
    test_collate_fn()
    print("✓ test_collate_fn passed.")
    
    test_text_regression_dataset()
    print("✓ test_text_regression_dataset passed.")
    
    # Registry tests
    test_encoder_registry()
    print("✓ test_encoder_registry passed.")
    
    test_model_registry()
    print("✓ test_model_registry passed.")
    
    test_loss_registry()
    print("✓ test_loss_registry passed.")
    
    # Encoder tests
    test_tfidf_encoder()
    print("✓ test_tfidf_encoder passed.")
    
    test_sentence_transformer_encoder()
    print("✓ test_sentence_transformer_encoder passed.")
    
    # Model tests
    test_lstm_model()
    print("✓ test_lstm_model passed.")
    
    test_gru_model()
    print("✓ test_gru_model passed.")
    
    # Estimator tests
    test_estimator_fit_predict_without_exogenous()
    print("✓ test_estimator_fit_predict_without_exogenous passed.")
    
    test_estimator_fit_predict_with_exogenous_and_early_stop()
    print("✓ test_estimator_fit_predict_with_exogenous_and_early_stop passed.")
    
    test_estimator_with_tfidf()
    print("✓ test_estimator_with_tfidf passed.")
    
    test_estimator_with_gru_model()
    print("✓ test_estimator_with_gru_model passed.")
    
    print("\n🎉 All tests passed for textregress 1.2.3!")
