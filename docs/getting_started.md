# Getting Started with TritonGNN

## Installation

Install TritonGNN from source:

```bash
git clone https://github.com/yourusername/tritongnn.git
cd tritongnn
pip install -e .
```

## Quick Start

### Basic Graph Creation

```python
import torch
from tritongnn import Graph

# Create a simple graph
x = torch.randn(5, 16)  # 5 nodes, 16 features each
edge_index = torch.tensor([
    [0, 1, 1, 2, 2, 3, 3, 4],  # source nodes
    [1, 0, 2, 1, 3, 2, 4, 3]   # target nodes
])

graph = Graph(x, edge_index)
print(graph)  # Graph(num_nodes=5, num_edges=8, num_features=16)
```

### Using GNN Layers

#### GCN (Graph Convolutional Network)

```python
from tritongnn import GCNConv

layer = GCNConv(in_features=16, out_features=32)
out = layer(graph.x, graph.edge_index)
print(out.shape)  # torch.Size([5, 32])
```

#### GAT (Graph Attention Network)

```python
from tritongnn import GATConv

layer = GATConv(
    in_features=16,
    out_features=32,
    heads=4,
    concat=True
)
out = layer(graph.x, graph.edge_index)
print(out.shape)  # torch.Size([5, 128]) - 4 heads * 32 features
```

#### GraphSAGE

```python
from tritongnn import SAGEConv

layer = SAGEConv(
    in_features=16,
    out_features=32,
    aggr='mean'
)
out = layer(graph.x, graph.edge_index)
```

#### GIN (Graph Isomorphism Network)

```python
import torch.nn as nn
from tritongnn import GINConv

mlp = nn.Sequential(
    nn.Linear(16, 32),
    nn.ReLU(),
    nn.Linear(32, 32)
)
layer = GINConv(mlp)
out = layer(graph.x, graph.edge_index)
```

### Building Multi-Layer Models

TritonGNN provides convenient multi-layer models:

```python
from tritongnn import GCN

model = GCN(
    in_features=16,
    hidden_features=64,
    out_features=7,  # 7 classes
    num_layers=3,
    dropout=0.5
)

out = model(graph.x, graph.edge_index)
print(out.shape)  # torch.Size([5, 7])
```

### Node Classification Example

```python
import torch.nn.functional as F
import torch.optim as optim
from tritongnn import GCN, load_cora, accuracy

# Load data
graph, train_mask, val_mask, test_mask = load_cora()
graph.add_self_loops()

# Create model
model = GCN(
    in_features=graph.num_features,
    hidden_features=64,
    out_features=7,
    num_layers=2
)

optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    
    out = model(graph.x, graph.edge_index)
    loss = F.cross_entropy(out[train_mask], graph.y[train_mask])
    
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        model.eval()
        with torch.no_grad():
            pred = model(graph.x, graph.edge_index).argmax(dim=-1)
            acc = accuracy(pred[val_mask], graph.y[val_mask])
            print(f"Epoch {epoch}, Val Acc: {acc:.4f}")
```

### Graph Classification Example

```python
from tritongnn import GIN, batch_graphs

# Create multiple graphs
graphs = [create_random_graph(20, 32, num_classes=2) for _ in range(10)]

# Batch them
batched = batch_graphs(graphs)

# Model with graph-level output
model = GIN(
    in_features=32,
    hidden_features=64,
    out_features=2,
    num_layers=5
)

# Forward pass with batch assignment
out = model(batched.x, batched.edge_index, batched.batch)
print(out.shape)  # torch.Size([10, 2]) - 10 graphs, 2 classes
```

## Key Features

### Triton Acceleration

TritonGNN uses custom Triton kernels for key operations:

- **Scatter operations**: scatter_add, scatter_mean, scatter_max
- **Message passing**: Optimized neighbor aggregation
- **Sparse operations**: Efficient SpMM for graph convolutions

These kernels provide significant speedups on GPU while maintaining PyTorch compatibility.

### Graph Data Structure

The `Graph` class provides a flexible container for graph data:

```python
graph = Graph(
    x=node_features,
    edge_index=edges,
    edge_attr=edge_features,  # optional
    y=labels,                  # optional
    batch=batch_assignment     # optional, for batching
)

# Useful methods
graph.add_self_loops()
graph.degree(mode='in')
graph.to(device)
```

### Batching Graphs

For graph-level tasks, batch multiple graphs:

```python
from tritongnn import batch_graphs

batched = batch_graphs([graph1, graph2, graph3])
# batched.batch contains assignment: [0,0,...,1,1,...,2,2,...]
```

## Next Steps

- Check out the `examples/` directory for complete examples
- Read the API reference for detailed documentation
- Explore custom layer implementations in `tritongnn/layers/`
- Learn about Triton kernels in `tritongnn/kernels/`

## Common Patterns

### Custom Message Passing Layer

```python
from tritongnn.core import MessagePassing

class MyLayer(MessagePassing):
    def __init__(self, in_feat, out_feat):
        super().__init__(aggr='add')
        self.lin = nn.Linear(in_feat, out_feat)
    
    def message(self, x_src, x_dst, edge_attr):
        # Custom message computation
        return self.lin(x_src)
    
    def update(self, aggr_out, x):
        # Custom update
        return F.relu(aggr_out)
```

### Using Edge Attributes

```python
# Create graph with edge features
edge_attr = torch.randn(num_edges, edge_dim)
graph = Graph(x, edge_index, edge_attr=edge_attr)

# Some layers can use edge attributes
out = layer(graph.x, graph.edge_index, graph.edge_attr)
```

## Troubleshooting

### CUDA Out of Memory

- Reduce batch size
- Use smaller hidden dimensions
- Enable gradient checkpointing
- Process graphs in smaller batches

### Slow Training

- Ensure CUDA is available: `torch.cuda.is_available()`
- Use mixed precision training
- Profile code to find bottlenecks
- Consider graph sampling for very large graphs