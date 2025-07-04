import pandas as pd
import numpy as np
import torch
import textregress
from textregress import TextRegressor
from textregress.models import LSTMTextRegressionModel, GRUTextRegressionModel
import os
import tempfile

os.environ['CURL_CA_BUNDLE'] = ''

print("# TextRegress Efficient Test Script\n")
print(f"TextRegress version: {getattr(textregress, '__version__', 'unknown')}")

# Prepare minimal test data
data = {
    "text": [
        "This is a test sentence.",
        "Another test sentence here.",
        "Third test sentence for validation."
    ],
    "y": [1.2, 2.3, 1.8],
    "ex1": [0.5, 1.0, 0.8],
    "ex2": [10, 20, 15]
}
df = pd.DataFrame(data)
print("\n## Data Preview\n", df.head())

# Test LSTM Model (minimal training)
print("\n## Test LSTM Model")
try:
    lstm_regressor = TextRegressor(
        model_name='lstm',
        exogenous_features=['ex1', 'ex2'],
        max_steps=5,  # Reduced steps
        early_stop_enabled=False
    )
    lstm_regressor.fit(df)
    lstm_preds = lstm_regressor.predict(df)
    print('LSTM predictions:', lstm_preds)
    print('✓ LSTM model training and prediction successful')
except Exception as e:
    print(f'✗ LSTM model failed: {e}')

# Test GRU Model (minimal training)
print("\n## Test GRU Model")
try:
    gru_regressor = TextRegressor(
        model_name='gru',
        exogenous_features=['ex1', 'ex2'],
        max_steps=5,  # Reduced steps
        early_stop_enabled=False
    )
    gru_regressor.fit(df)
    gru_preds = gru_regressor.predict(df)
    print('GRU predictions:', gru_preds)
    print('✓ GRU model training and prediction successful')
except Exception as e:
    print(f'✗ GRU model failed: {e}')

# Test fit_predict method
print("\n## Test fit_predict method")
try:
    fit_pred_regressor = TextRegressor(
        model_name='lstm',
        exogenous_features=['ex1', 'ex2'],
        max_steps=3,  # Very minimal training
        early_stop_enabled=False
    )
    fit_preds = fit_pred_regressor.fit_predict(df)
    print('fit_predict results:', fit_preds)
    print('✓ fit_predict method successful')
except Exception as e:
    print(f'✗ fit_predict failed: {e}')

# Test embedding extraction (simplified)
print("\n## Test Embedding Extraction")
try:
    # Use the trained model
    lstm_model = lstm_regressor.model
    
    # Create simple test input
    test_text = "Simple test text for embedding."
    test_exogenous = torch.tensor([[0.5, 10.0]], dtype=torch.float32)
    
    # Get embeddings using the model's methods
    doc_emb = lstm_model.get_document_embedding(
        torch.randn(1, 10, 384),  # Simple random input
        test_exogenous
    )
    print('Document embedding shape:', doc_emb.shape)
    print('✓ Embedding extraction successful')
except Exception as e:
    print(f'✗ Embedding extraction failed: {e}')

# Test model save/load (simplified)
print("\n## Test Model Save/Load")
try:
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp_file:
        model_path = tmp_file.name
    
    lstm_model.save(model_path)
    loaded_model = LSTMTextRegressionModel.load(model_path)
    os.remove(model_path)
    
    # Test loaded model
    test_output = loaded_model(
        torch.randn(1, 10, 384),
        torch.tensor([[0.5, 10.0]], dtype=torch.float32)
    )
    print('Loaded model output shape:', test_output.shape)
    print('✓ Model save/load successful')
except Exception as e:
    print(f'✗ Model save/load failed: {e}')

print("\n## Test Summary")
print("All core functionality tests completed!") 
