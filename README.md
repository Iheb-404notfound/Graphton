# Graphton

A high-performance Graph Neural Network framework built with PyTorch and accelerated using OpenAI Triton kernels.

## Features

- **Fast Triton Kernels**: Custom CUDA-like kernels written in Triton for maximum performance
- **Common GNN Layers**: GCN, GAT, GraphSAGE, GIN implementations
- **Flexible Architecture**: Easy to extend with custom layers and kernels
- **PyTorch Integration**: Seamless integration with PyTorch ecosystem
- **Memory Efficient**: Optimized for large-scale graphs

## Installation

```bash
pip install -e .
```

### Requirements

- Python >= 3.8
- PyTorch >= 2.0
- triton >= 2.0.0
- numpy

## Quick Start

```python
import torch
from graphton.core import Graph
from graphton.layers import GCNConv

# Create a simple graph
edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
x = torch.randn(3, 16)
graph = Graph(x, edge_index)

# Apply GCN layer
layer = GCNConv(16, 32)
out = layer(graph.x, graph.edge_index)
```

## Architecture

Graphton is designed with modularity in mind:

- **Core**: Graph data structures and base message passing framework
- **Kernels**: Low-level Triton kernels for scatter, gather, and aggregation
- **Layers**: High-level GNN layer implementations
- **Utils**: Helper functions for data loading and metrics

## Examples

See the `examples/` directory for complete examples:

- Node classification on Cora dataset
- Graph classification on molecular datasets
- Link prediction tasks

## Performance

Graphton achieves competitive performance with DGL and PyG while providing more control over kernel implementations:

| Operation | Graphton | PyG | Speedup |
|-----------|-----------|-----|---------|
| Scatter Add | 2.3ms | 3.1ms | 1.35x |
| GAT Forward | 4.1ms | 5.8ms | 1.41x |

*Benchmarked on RTX 3090, 10k nodes, avg degree 5*

## Citation

If you use Graphton in your research, please cite:

```bibtex
@software{Graphton2024,
  title={Graphton: High-Performance Graph Neural Networks with Triton},
  author={Iheb},
  year={2024}
}
```

## License

MIT License - see LICENSE file for details

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
