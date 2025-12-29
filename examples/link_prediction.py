"""
Link Prediction Example using GraphSAGE.

This example demonstrates:
- Edge sampling for link prediction
- Using node embeddings to predict links
- Training with binary cross-entropy loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from graphton import GraphSAGE, create_random_graph


class LinkPredictor(nn.Module):
    """
    Link prediction model combining GNN encoder with edge decoder.
    """
    
    def __init__(self, in_features, hidden_features, num_layers=2):
        super().__init__()
        
        # GNN encoder
        self.encoder = GraphSAGE(
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=hidden_features,
            num_layers=num_layers,
            dropout=0.0,
            aggr='mean'
        )
        
        # Edge decoder
        self.decoder = nn.Sequential(
            nn.Linear(2 * hidden_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, 1)
        )
    
    def encode(self, x, edge_index):
        """Encode nodes to embeddings."""
        return self.encoder(x, edge_index)
    
    def decode(self, z, edge_index):
        """Decode edge probabilities from node embeddings."""
        # Get source and target embeddings
        src_emb = z[edge_index[0]]
        dst_emb = z[edge_index[1]]
        
        # Concatenate and predict
        edge_emb = torch.cat([src_emb, dst_emb], dim=-1)
        return self.decoder(edge_emb).squeeze(-1)
    
    def forward(self, x, edge_index, edge_label_index):
        """
        Forward pass.
        
        Args:
            x: Node features
            edge_index: Training edges
            edge_label_index: Edges to predict
        """
        z = self.encode(x, edge_index)
        return self.decode(z, edge_label_index)


def negative_sampling(edge_index, num_nodes, num_neg_samples):
    """
    Sample negative edges (non-existing edges).
    
    Args:
        edge_index: Positive edges [2, num_edges]
        num_nodes: Number of nodes
        num_neg_samples: Number of negative samples
        
    Returns:
        Negative edge indices [2, num_neg_samples]
    """
    # Create set of existing edges
    edge_set = set()
    for i in range(edge_index.size(1)):
        src, dst = edge_index[0, i].item(), edge_index[1, i].item()
        edge_set.add((src, dst))
        edge_set.add((dst, src))  # Undirected
    
    # Sample negative edges
    neg_edges = []
    while len(neg_edges) < num_neg_samples:
        src = torch.randint(0, num_nodes, (1,)).item()
        dst = torch.randint(0, num_nodes, (1,)).item()
        
        if src != dst and (src, dst) not in edge_set:
            neg_edges.append([src, dst])
    
    return torch.tensor(neg_edges, dtype=torch.long).t()


def train(model, graph, optimizer, train_edge_index, device):
    """Train for one epoch."""
    model.train()
    optimizer.zero_grad()
    
    # Positive edges
    pos_edge_index = train_edge_index
    
    # Negative sampling
    neg_edge_index = negative_sampling(
        pos_edge_index, 
        graph.num_nodes, 
        pos_edge_index.size(1)
    ).to(device)
    
    # Combine positive and negative edges
    edge_label_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
    edge_labels = torch.cat([
        torch.ones(pos_edge_index.size(1)),
        torch.zeros(neg_edge_index.size(1))
    ]).to(device)
    
    # Forward pass
    out = model(graph.x, train_edge_index, edge_label_index)
    
    # Compute loss
    loss = F.binary_cross_entropy_with_logits(out, edge_labels)
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    return loss.item()


@torch.no_grad()
def evaluate(model, graph, test_edge_index, device):
    """Evaluate the model."""
    model.eval()
    
    # Positive edges
    pos_edge_index = test_edge_index
    
    # Negative sampling
    neg_edge_index = negative_sampling(
        pos_edge_index,
        graph.num_nodes,
        pos_edge_index.size(1)
    ).to(device)
    
    # Predictions
    pos_pred = torch.sigmoid(model(graph.x, graph.edge_index, pos_edge_index))
    neg_pred = torch.sigmoid(model(graph.x, graph.edge_index, neg_edge_index))
    
    # Compute AUC (simplified)
    pred = torch.cat([pos_pred, neg_pred])
    labels = torch.cat([
        torch.ones(pos_pred.size(0)),
        torch.zeros(neg_pred.size(0))
    ])
    
    # Sort by prediction
    sorted_indices = torch.argsort(pred, descending=True)
    sorted_labels = labels[sorted_indices]
    
    # Compute AUC
    num_pos = (sorted_labels == 1).sum().item()
    num_neg = (sorted_labels == 0).sum().item()
    
    fp = tp = 0
    auc = 0.0
    
    for label in sorted_labels:
        if label == 1:
            tp += 1
        else:
            fp += 1
            auc += tp
    
    auc /= (num_pos * num_neg) if num_pos * num_neg > 0 else 1
    
    return auc


def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create random graph
    print("Creating random graph...")
    graph = create_random_graph(
        num_nodes=500,
        num_features=64,
        avg_degree=10
    )
    graph = graph.to(device)
    
    print(f"Graph: {graph}")
    
    # Split edges into train/test
    num_edges = graph.num_edges
    perm = torch.randperm(num_edges)
    train_size = int(0.8 * num_edges)
    
    train_edge_index = graph.edge_index[:, perm[:train_size]]
    test_edge_index = graph.edge_index[:, perm[train_size:]]
    
    print(f"Train edges: {train_size}, Test edges: {num_edges - train_size}")
    
    # Create model
    model = LinkPredictor(
        in_features=graph.num_features,
        hidden_features=128,
        num_layers=2
    ).to(device)
    
    print(f"\nModel architecture:")
    print(model)
    print(f"Parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    print("\nTraining...")
    
    for epoch in range(1, 101):
        loss = train(model, graph, optimizer, train_edge_index, device)
        
        if epoch % 10 == 0:
            train_auc = evaluate(model, graph, train_edge_index, device)
            test_auc = evaluate(model, graph, test_edge_index, device)
            
            print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | "
                  f"Train AUC: {train_auc:.4f} | Test AUC: {test_auc:.4f}")
    
    # Final test evaluation
    final_test_auc = evaluate(model, graph, test_edge_index, device)
    print(f"\nFinal Test AUC: {final_test_auc:.4f}")


if __name__ == '__main__':
    main()