"""
Graphton: High-Performance Graph Neural Networks with Triton
"""

__version__ = '0.1.0'

from .core.graph import Graph, batch_graphs
from .core.message_passing import MessagePassing

from .layers.gcn import GCNConv, GCN
from .layers.gat import GATConv
from .layers.sage import SAGEConv, GraphSAGE
from .layers.gin import GINConv, GIN

from .utils.data import (
    load_cora,
    create_random_graph,
    create_batch_loader,
    add_self_loops_and_normalize,
    k_hop_subgraph,
    to_undirected,
)

from .utils.metrics import (
    accuracy,
    f1_score,
    cross_entropy_loss,
    mean_squared_error,
    mean_absolute_error,
    auc_score,
    Evaluator,
)

__all__ = [
    # Core
    'Graph',
    'batch_graphs',
    'MessagePassing',
    
    # Layers
    'GCNConv',
    'GCN',
    'GATConv',
    'SAGEConv',
    'GraphSAGE',
    'GINConv',
    'GIN',
    
    # Utils
    'load_cora',
    'create_random_graph',
    'create_batch_loader',
    'add_self_loops_and_normalize',
    'k_hop_subgraph',
    'to_undirected',
    
    # Metrics
    'accuracy',
    'f1_score',
    'cross_entropy_loss',
    'mean_squared_error',
    'mean_absolute_error',
    'auc_score',
    'Evaluator',
]