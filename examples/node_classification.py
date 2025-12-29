"""
Node Classification Example using GCN on Cora dataset.

This example demonstrates:
- Loading graph data
- Creating a GCN model
- Training with cross-entropy loss
- Evaluating on test set
"""

import torch
import torch.nn.functional as F
import torch.optim as optim

from graphton import GCN, load_cora, accuracy


def train(model, graph, optimizer, train_mask):
    """Train for one epoch."""
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    out = model(graph.x, graph.edge_index)
    
    # Compute loss only on training nodes
    loss = F.cross_entropy(out[train_mask], graph.y[train_mask])
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    return loss.item()


@torch.no_grad()
def evaluate(model, graph, mask):
    """Evaluate the model."""
    model.eval()
    out = model(graph.x, graph.edge_index)
    pred = out.argmax(dim=-1)
    
    acc = accuracy(pred[mask], graph.y[mask])
    
    return acc


def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("Loading Cora dataset...")
    graph, train_mask, val_mask, test_mask = load_cora()
    graph = graph.to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)
    
    print(f"Graph: {graph}")
    print(f"Train/Val/Test: {train_mask.sum()}/{val_mask.sum()}/{test_mask.sum()}")
    
    # Add self-loops
    graph.add_self_loops()
    
    # Create model
    num_features = graph.num_features
    num_classes = graph.y.max().item() + 1
    
    model = GCN(
        in_features=num_features,
        hidden_features=64,
        out_features=num_classes,
        num_layers=2,
        dropout=0.5
    ).to(device)
    
    print(f"\nModel: {model}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    
    # Training loop
    print("\nTraining...")
    best_val_acc = 0
    
    for epoch in range(1, 201):
        loss = train(model, graph, optimizer, train_mask)
        
        if epoch % 10 == 0:
            train_acc = evaluate(model, graph, train_mask)
            val_acc = evaluate(model, graph, val_mask)
            
            print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | "
                  f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
    
    # Test evaluation
    test_acc = evaluate(model, graph, test_mask)
    print(f"\nTest Accuracy: {test_acc:.4f}")


if __name__ == '__main__':
    main()