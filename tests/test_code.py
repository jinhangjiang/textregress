import pandas as pd
import torch
import numpy as np

# --- Import functions/classes from the package ---
from textregress.encoding import (
    get_encoder,
    SentenceTransformerEncoder,
    TfidfEncoder,
    HuggingFaceEncoder
)
from textregress.utils import chunk_text, pad_chunks, collate_fn, TextRegressionDataset
from textregress.estimator import TextRegressor

# ---------------------------------
# Utility Function Tests (utils.py)
# ---------------------------------
def test_chunk_text():
    text = "one two three four five six seven eight nine"
    # Using chunk_info=(3, 1) should yield only full chunks.
    chunks = chunk_text(text, chunk_info=(3, 1), encoder=None)  # encoder not used in chunk_text
    expected = ["one two three", "three four five", "five six seven", "seven eight nine"]
    assert chunks == expected, f"Expected {expected}, got {chunks}"

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
# Encoder Tests (encoding.py)
# ---------------------------------
def test_tfidf_encoder():
    encoder = TfidfEncoder()
    texts = ["hello world", "test sentence"]
    encoder.fit(texts)
    vector = encoder.encode("hello world")
    # Expect a 1D numpy array.
    assert isinstance(vector, np.ndarray)
    assert vector.ndim == 1

def test_sentence_transformer_encoder():
    # This test loads the actual model; may take a few seconds.
    encoder = SentenceTransformerEncoder("sentence-transformers/all-MiniLM-L6-v2")
    vector = encoder.encode("Hello world")
    assert torch.is_tensor(vector)
    # Expected output dimension for all-MiniLM-L6-v2 is 384.
    assert vector.shape[-1] == 384, f"Expected 384, got {vector.shape[-1]}"

def test_huggingface_encoder():
    encoder = HuggingFaceEncoder("bert-base-uncased")
    vector = encoder.encode("Hello world")
    assert torch.is_tensor(vector)
    # bert-base-uncased typically produces embeddings of dimension 768.
    assert vector.shape[-1] == 768, f"Expected 768, got {vector.shape[-1]}"

# ---------------------------------
# Estimator Integration Tests (estimator.py)
# ---------------------------------
# Create a dummy encoder for fast testing.
class DummyEncoder:
    def encode(self, text):
        # Return a fixed tensor of ones with dimension 768.
        return torch.ones(768)

def test_estimator_fit_predict_without_exogenous():
    df = pd.DataFrame({
        "text": ["Test sentence one.", "Test sentence two."],
        "y": [1.0, 2.0]
    })
    # Monkey-patch get_encoder to return DummyEncoder.
    from textregress import encoding
    original_get_encoder = encoding.get_encoder
    encoding.get_encoder = lambda model_id: DummyEncoder()
    
    estimator = TextRegressor(
        encoder_model="sentence-transformers/all-MiniLM-L6-v2",
        max_steps=2,
        early_stop_enabled=False,
        loss_function="mae"
    )
    estimator.fit(df)
    preds = estimator.predict(df)
    assert isinstance(preds, list)
    assert len(preds) == len(df)
    for pred in preds:
        assert isinstance(pred, float)
    
    # Restore original function.
    encoding.get_encoder = original_get_encoder

def test_estimator_fit_predict_with_exogenous_and_early_stop():
    df = pd.DataFrame({
        "text": ["Exogenous test one.", "Exogenous test two.", "Exogenous test three."],
        "y": [1.0, 2.0, 3.0],
        "exo1": [0.1, 0.2, 0.3],
        "exo2": [10, 20, 30]
    })
    from textregress import encoding
    original_get_encoder = encoding.get_encoder
    encoding.get_encoder = lambda model_id: DummyEncoder()
    
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
    assert isinstance(preds, list)
    assert len(preds) == len(df)
    for pred in preds:
        assert isinstance(pred, float)
    
    encoding.get_encoder = original_get_encoder

# --------------------------------
# Run Tests if This Script is Main
# --------------------------------
if __name__ == "__main__":
    print("Running tests...")
    
    test_chunk_text()
    print("test_chunk_text passed.")
    
    test_pad_chunks()
    print("test_pad_chunks passed.")
    
    test_collate_fn()
    print("test_collate_fn passed.")
    
    test_text_regression_dataset()
    print("test_text_regression_dataset passed.")
    
    test_tfidf_encoder()
    print("test_tfidf_encoder passed.")
    
    test_sentence_transformer_encoder()
    print("test_sentence_transformer_encoder passed.")
    
    test_huggingface_encoder()
    print("test_huggingface_encoder passed.")
    
    test_estimator_fit_predict_without_exogenous()
    print("test_estimator_fit_predict_without_exogenous passed.")
    
    test_estimator_fit_predict_with_exogenous_and_early_stop()
    print("test_estimator_fit_predict_with_exogenous_and_early_stop passed.")
    
    print("All tests passed!")
