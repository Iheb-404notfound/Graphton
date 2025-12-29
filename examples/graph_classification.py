"""
Graph Classification Example using GIN.

This example demonstrates:
- Creating multiple graphs
- Batching graphs together
- Using GIN for graph-level predictions
- Training a graph classifier
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from graphton import GIN, create_random_graph, create_batch_loader, accuracy


def generate_synthetic_dataset(num_graphs=200, num_nodes_range=(10, 50)):
    """Generate synthetic graph classification dataset."""
    graphs = []
    
    for i in range(num_graphs):
        # Random graph size
        num_nodes = torch.randint(*num_nodes_range, (1,)).item()
        
        # Create graph with 2 classes
        graph = create_random_graph(
            num_nodes=num_nodes,
            num_features=32,
            avg_degree=5,
            num_classes=2
        )
        
        # Graph label (binary classification)
        graph.y = torch.randint(0, 2, (1,))
        
        graphs.append(graph)
    
    return graphs


def train_epoch(model, loader, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        out = model(batch.x, batch.edge_index, batch.batch)
        
        # Compute loss
        loss = F.cross_entropy(out, batch.y)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item() * batch.y.size(0)
        pred = out.argmax(dim=-1)
        total_correct += (pred == batch.y).sum().item()
        total_samples += batch.y.size(0)
    
    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def evaluate(model, loader, device):
    """Evaluate the model."""
    model.eval()
    total_correct = 0
    total_samples = 0
    
    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.batch)
        pred = out.argmax(dim=-1)
        
        total_correct += (pred == batch.y).sum().item()
        total_samples += batch.y.size(0)
    
    return total_correct / total_samples


def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Generate synthetic dataset
    print("Generating synthetic dataset...")
    all_graphs = generate_synthetic_dataset(num_graphs=400)
    
    # Split into train/val/test
    train_graphs = all_graphs[:240]
    val_graphs = all_graphs[240:320]
    test_graphs = all_graphs[320:]
    
    print(f"Train/Val/Test: {len(train_graphs)}/{len(val_graphs)}/{len(test_graphs)}")
    
    # Create data loaders
    batch_size = 32
    train_loader = create_batch_loader(train_graphs, batch_size)
    val_loader = create_batch_loader(val_graphs, batch_size)
    test_loader = create_batch_loader(test_graphs, batch_size)
    
    # Create model
    model = GIN(
        in_features=32,
        hidden_features=64,
        out_features=2,  # Binary classification
        num_layers=5,
        dropout=0.5
    ).to(device)
    
    print(f"\nModel: {model}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    print("\nTraining...")
    best_val_acc = 0
    
    for epoch in range(1, 101):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        
        if epoch % 10 == 0:
            val_acc = evaluate(model, val_loader, device)
            
            print(f"Epoch {epoch:03d} | Loss: {train_loss:.4f} | "
                  f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
    
    # Test evaluation
    test_acc = evaluate(model, test_loader, device)
    print(f"\nTest Accuracy: {test_acc:.4f}")


if __name__ == '__main__':
    main()