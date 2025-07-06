#!/usr/bin/env python3
"""
Simple example showing how to use the new get_feature_importance method.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from textregress import TextRegressor

# Create sample data
data = {
    'text': [
        "This is a positive review about the product quality.",
        "The customer service was excellent and helpful.",
        "Not satisfied with the purchase experience.",
        "Great value for money and fast delivery.",
        "Poor quality product, would not recommend."
    ],
    'y': [4.5, 4.8, 2.1, 4.2, 1.5],
    'feature1': [1.0, 1.2, 0.8, 1.1, 0.7],
    'feature2': [0.5, 0.6, 0.3, 0.7, 0.2]
}
df = pd.DataFrame(data)

# Create and fit the model
regressor = TextRegressor(
    model_name="lstm",
    encoder_model="sentence-transformers/all-MiniLM-L6-v2",
    exogenous_features=["feature1", "feature2"],
    max_steps=20,
    learning_rate=0.001,
    batch_size=2
)

print("Fitting model...")
regressor.fit(df)
print("Model fitted successfully!")

# Get feature importance - it's that simple!
print("\nGetting feature importance...")

# Method 1: Get importance for training data (default)
importance = regressor.get_feature_importance()
print(f"Text importance shape: {importance['text_importance'].shape}")
print(f"Exogenous importance shape: {importance['exogenous_importance'].shape}")

# Method 2: Get importance for specific data
test_df = df.head(2)
test_importance = regressor.get_feature_importance(test_df)
print(f"Test data importance shape: {test_importance['text_importance'].shape}")

# Method 3: Get attention weights (if cross-attention is enabled)
regressor_attention = TextRegressor(
    model_name="lstm",
    encoder_model="sentence-transformers/all-MiniLM-L6-v2",
    exogenous_features=["feature1", "feature2"],
    cross_attention_enabled=True,
    max_steps=20,
    learning_rate=0.001,
    batch_size=2
)

print("\nFitting model with cross-attention...")
regressor_attention.fit(df)

attention_importance = regressor_attention.get_feature_importance(mode="attention")
print(f"Attention importance shape: {attention_importance['text_importance'].shape}")

# Simple visualization
def plot_importance(importance, title):
    """Simple plot of feature importance."""
    plt.figure(figsize=(12, 6))
    
    # Plot text importance
    plt.subplot(1, 2, 1)
    text_imp = importance['text_importance']
    plt.imshow(text_imp, cmap='Reds', aspect='auto')
    plt.colorbar(label='Importance Score')
    plt.title(f'{title} - Text Importance')
    plt.xlabel('Word Position')
    plt.ylabel('Sample')
    
    # Plot exogenous importance
    if 'exogenous_importance' in importance:
        plt.subplot(1, 2, 2)
        exo_imp = importance['exogenous_importance']
        feature_names = ["feature1", "feature2"]
        plt.bar(feature_names, exo_imp.mean(axis=0))
        plt.title(f'{title} - Exogenous Features')
        plt.ylabel('Average Importance')
    
    plt.tight_layout()
    plt.show()

# Plot the results
plot_importance(importance, "Gradient Importance")
plot_importance(attention_importance, "Attention Importance")

print("\nFeature importance analysis completed!")
print("\nKey benefits of the new API:")
print("1. Simple: regressor.get_feature_importance()")
print("2. Flexible: Can analyze training data or new data")
print("3. Multiple modes: gradient or attention")
print("4. Clean output: numpy arrays ready for analysis") 