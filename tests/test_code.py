from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

import textregress
import textregress.estimator as estimator_module
from textregress import TextRegressor
from textregress.encoders import get_encoder, list_available_encoders
from textregress.losses import get_loss_function, list_available_losses
from textregress.models import get_model, list_available_models
from textregress.utils import (
    TextRegressionDataset,
    chunk_text,
    collate_fn,
    integrated_gradients,
    pad_chunks,
)


class DummyEncoder:
    def __init__(self, output_dim: int = 8):
        self.fitted = True
        self.output_dim = output_dim

    def encode(self, text):
        length = float(len(text.split()) or 1)
        return torch.full((self.output_dim,), length, dtype=torch.float32)

    def fit(self, texts):
        self.fitted = True


def test_chunk_text():
    text = "one two three four five six seven eight nine"
    assert chunk_text(text, max_length=3, overlap=1) == [
        "one two three",
        "three four five",
        "five six seven",
        "seven eight nine",
    ]
    assert chunk_text(text, max_length=3, overlap=0) == [
        "one two three",
        "four five six",
        "seven eight nine",
    ]


def test_pad_chunks():
    chunks = ["a", "b", "c"]
    assert pad_chunks(chunks, padding_value=0) == chunks


def test_collate_fn_without_exogenous():
    batch = [
        ([torch.ones(4), torch.ones(4) * 2], 1.0),
        ([torch.ones(4) * 3], 2.0),
    ]

    padded_sequences, targets = collate_fn(batch)

    assert padded_sequences.shape == (2, 2, 4)
    assert torch.equal(targets, torch.tensor([1.0, 2.0]))


def test_text_regression_dataset():
    dataset = TextRegressionDataset(
        encoded_sequences=[[torch.zeros(5), torch.ones(5)]],
        targets=[10.0],
    )

    sample, target = dataset[0]

    assert len(dataset) == 1
    assert isinstance(sample, list)
    assert sample[0].shape == (5,)
    assert target == 10.0


def test_registry_exports():
    assert "sentence_transformer" in list_available_encoders()
    assert "tfidf" in list_available_encoders()
    assert "lstm" in list_available_models()
    assert "gru" in list_available_models()
    assert "mae" in list_available_losses()
    assert "mse" in list_available_losses()
    assert get_loss_function("mae") is not None


def test_tfidf_encoder():
    encoder = get_encoder("tfidf")
    encoder.fit(["hello world", "another test"])

    vector = encoder.encode("hello world")

    assert torch.is_tensor(vector)
    assert vector.ndim == 1
    assert vector.shape[0] > 0


@pytest.mark.parametrize("model_name", ["lstm", "gru"])
def test_model_registry_instances(model_name):
    model_cls = get_model(model_name)
    model = model_cls(
        encoder_output_dim=8,
        learning_rate=1e-3,
        loss_function="mae",
    )

    x = torch.randn(2, 3, 8)
    output = model(x)

    assert output.shape == (2, 1)


def test_integrated_gradients_returns_expected_keys():
    model = get_model("lstm")(
        encoder_output_dim=8,
        learning_rate=1e-3,
        loss_function="mae",
        exogenous_features=["feature1", "feature2"],
    )

    x = torch.randn(2, 3, 8)
    exogenous = torch.randn(2, 2)

    result = integrated_gradients(model, x, exogenous, steps=4)

    assert "text_importance" in result
    assert result["text_importance"].shape == (2, 3)
    assert "exogenous_importance" in result
    assert result["exogenous_importance"].shape == (2, 2)


def test_estimator_fit_predict_with_tfidf():
    df = pd.DataFrame(
        {
            "text": ["TF-IDF test one.", "TF-IDF test two."],
            "y": [1.0, 2.0],
        }
    )

    estimator = TextRegressor(
        encoder_model="tfidf",
        max_steps=1,
        early_stop_enabled=False,
        loss_function="mae",
        hidden_size=16,
        inference_layer_units=8,
    )

    predictions = estimator.fit_predict(df, batch_size=2)

    assert isinstance(predictions, np.ndarray)
    assert len(predictions) == len(df)


def test_estimator_save_and_load_round_trip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(estimator_module, "get_encoder", lambda *args, **kwargs: DummyEncoder())

    df = pd.DataFrame(
        {
            "text": ["one short document", "another short document"],
            "y": [1.0, 2.0],
        }
    )

    estimator = TextRegressor(
        encoder_model="sentence-transformers/all-MiniLM-L6-v2",
        max_steps=1,
        early_stop_enabled=False,
        loss_function="mae",
        hidden_size=16,
        inference_layer_units=8,
    )
    estimator.fit(df, batch_size=2)

    save_path = tmp_path / "regressor.pkl"
    estimator.save(save_path)

    loaded = TextRegressor.load(save_path, device="cpu")
    predictions = loaded.predict(df, batch_size=2)

    assert save_path.exists()
    assert loaded.model is not None
    assert loaded.encoder is not None
    assert isinstance(predictions, np.ndarray)
    assert len(predictions) == len(df)


def test_version():
    assert hasattr(textregress, "__version__")
    assert textregress.__version__ == "1.3.0"
