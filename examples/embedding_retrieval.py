"""
Example: Retrieving document embeddings from trained models.

This example demonstrates how to:
1. Train a text regression model
2. Extract document embeddings for transfer learning
3. Use embeddings for downstream tasks
"""

import torch
import pandas as pd
from textregress.models import get_model
from textregress.encoders import get_encoder
from textregress.utils import TextRegressionDataset
from textregress.losses import get_loss_function
from torch.utils.data import DataLoader
import numpy as np


def create_sample_data():
    """Create sample data for demonstration."""
    texts = [
        "This is a positive review about a great product.",
        "The quality of this item is excellent and I highly recommend it.",
        "This product exceeded my expectations and works perfectly.",
        "I'm very satisfied with this purchase and would buy again.",
        "The customer service was outstanding and the product is amazing.",
        "This is a negative review about a terrible product.",
        "The quality is poor and I would not recommend this item.",
        "This product failed to meet my expectations and broke quickly.",
        "I'm very disappointed with this purchase and regret buying it.",
        "The customer service was awful and the product is defective."
    ]
    
    # Create regression targets (e.g., sentiment scores)
    targets = [4.5, 4.8, 4.2, 4.6, 4.9, 1.2, 1.5, 1.8, 1.1, 1.3]
    
    return pd.DataFrame({
        'text': texts,
        'target': targets
    })


def train_model(df):
    """Train a simple model for demonstration."""
    # Initialize encoder
    encoder = get_encoder("sentence_transformer", model_name="all-MiniLM-L6-v2")
    
    # Create dataset
    dataset = TextRegressionDataset(
        texts=df['text'].tolist(),
        targets=df['target'].tolist(),
        encoder=encoder
    )
    
    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # Initialize model
    model = get_model("lstm")(
        encoder_output_dim=384,  # all-MiniLM-L6-v2 dimension
        hidden_size=128,
        rnn_layers=1,
        inference_layer_units=64,
        learning_rate=0.001
    )
    
    # Simple training loop (for demonstration)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = get_loss_function("mse")
    
    model.train()
    for epoch in range(5):
        total_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            
            x = batch['text']
            y = batch['target']
            
            # Forward pass
            y_pred = model(x)
            loss = loss_fn(y_pred.squeeze(), y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")
    
    return model, encoder


def extract_embeddings(model, encoder, texts):
    """Extract document embeddings from the trained model."""
    model.eval()
    
    # Encode texts
    encoded_texts = []
    for text in texts:
        encoded = encoder.encode(text)
        if isinstance(encoded, list):
            encoded = torch.tensor(encoded)
        encoded_texts.append(encoded)
    
    # Stack encoded texts
    x = torch.stack(encoded_texts).unsqueeze(0)  # Add batch dimension
    
    with torch.no_grad():
        # Get sequence embeddings
        sequence_embeddings = model.get_sequence_embeddings(x)
        print(f"Sequence embeddings shape: {sequence_embeddings.shape}")
        
        # Get document embeddings
        document_embeddings = model.get_document_embedding(x)
        print(f"Document embeddings shape: {document_embeddings.shape}")
        
        # Get predictions
        predictions = model(x)
        print(f"Predictions shape: {predictions.shape}")
    
    return {
        'sequence_embeddings': sequence_embeddings.squeeze(0),  # Remove batch dimension
        'document_embeddings': document_embeddings.squeeze(0),
        'predictions': predictions.squeeze()
    }


def use_embeddings_for_downstream_task(embeddings, labels):
    """Demonstrate using embeddings for a downstream task (e.g., clustering)."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    
    # Use document embeddings for clustering
    doc_embeddings = embeddings['document_embeddings'].numpy()
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=2, random_state=42)
    cluster_labels = kmeans.fit_predict(doc_embeddings)
    
    # Calculate silhouette score
    silhouette_avg = silhouette_score(doc_embeddings, cluster_labels)
    print(f"Silhouette Score: {silhouette_avg:.4f}")
    
    # Analyze clusters
    for i in range(2):
        cluster_indices = np.where(cluster_labels == i)[0]
        cluster_labels_actual = [labels[j] for j in cluster_indices]
        print(f"Cluster {i}: {len(cluster_indices)} samples, "
              f"avg sentiment: {np.mean(cluster_labels_actual):.2f}")


def main():
    """Main function demonstrating embedding retrieval."""
    print("=== TextRegress Embedding Retrieval Example ===\n")
    
    # Create sample data
    print("1. Creating sample data...")
    df = create_sample_data()
    print(f"Created dataset with {len(df)} samples\n")
    
    # Train model
    print("2. Training model...")
    model, encoder = train_model(df)
    print("Training completed!\n")
    
    # Extract embeddings
    print("3. Extracting embeddings...")
    new_texts = [
        "This is a wonderful product that I love!",
        "Terrible quality, would not recommend.",
        "Amazing customer service and great product.",
        "Poor experience, very disappointed."
    ]
    
    embeddings = extract_embeddings(model, encoder, new_texts)
    print("Embedding extraction completed!\n")
    
    # Use embeddings for downstream task
    print("4. Using embeddings for clustering...")
    new_labels = [4.5, 1.2, 4.8, 1.5]  # Example sentiment scores
    use_embeddings_for_downstream_task(embeddings, new_labels)
    
    print("\n=== Example completed! ===")
    print("\nKey takeaways:")
    print("- Use get_sequence_embeddings() for sequence-level representations")
    print("- Use get_document_embedding() for document-level representations")
    print("- Embeddings can be used for transfer learning, clustering, etc.")
    print("- The model maintains the same interface for both prediction and embedding extraction")


if __name__ == "__main__":
    main() 