import torch
import numpy as np
from typing import Tuple, Optional
from ..core.graph import Graph


def load_cora() -> Tuple[Graph, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create a synthetic Cora-like dataset for demonstration.
    
    In practice, you would load the actual Cora dataset.
    
    Returns:
        graph: Graph object
        train_mask: Training node mask
        val_mask: Validation node mask
        test_mask: Test node mask
    """
    # Synthetic data (replace with actual data loading)
    num_nodes = 2708
    num_features = 1433
    num_classes = 7
    num_edges = 5429
    
    # Random features
    x = torch.randn(num_nodes, num_features)
    
    # Random edges
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    
    # Random labels
    y = torch.randint(0, num_classes, (num_nodes,))
    
    # Create masks (60/20/20 split)
    indices = torch.randperm(num_nodes)
    train_size = int(0.6 * num_nodes)
    val_size = int(0.2 * num_nodes)
    
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    train_mask[indices[:train_size]] = True
    val_mask[indices[train_size:train_size+val_size]] = True
    test_mask[indices[train_size+val_size:]] = True
    
    graph = Graph(x, edge_index, y=y)
    
    return graph, train_mask, val_mask, test_mask


def create_random_graph(
    num_nodes: int,
    num_features: int,
    avg_degree: int = 5,
    num_classes: Optional[int] = None
) -> Graph:
    """
    Create a random graph for testing.
    
    Args:
        num_nodes: Number of nodes
        num_features: Number of node features
        avg_degree: Average node degree
        num_classes: Number of classes for labels (optional)
        
    Returns:
        Graph object
    """
    # Generate random features
    x = torch.randn(num_nodes, num_features)
    
    # Generate random edges (Erdős–Rényi model)
    num_edges = num_nodes * avg_degree
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    
    # Remove self-loops
    mask = edge_index[0] != edge_index[1]
    edge_index = edge_index[:, mask]
    
    # Generate random labels if requested
    y = None
    if num_classes is not None:
        y = torch.randint(0, num_classes, (num_nodes,))
    
    return Graph(x, edge_index, y=y)


def create_batch_loader(graphs: list, batch_size: int):
    """
    Create a simple batch loader for graphs.
    
    Args:
        graphs: List of Graph objects
        batch_size: Batch size
        
    Yields:
        Batched graphs
    """
    from ..core.graph import batch_graphs
    
    for i in range(0, len(graphs), batch_size):
        batch = graphs[i:i+batch_size]
        yield batch_graphs(batch)


def add_self_loops_and_normalize(
    edge_index: torch.Tensor,
    num_nodes: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Add self-loops and compute normalized edge weights for GCN.
    
    Args:
        edge_index: Edge connectivity [2, num_edges]
        num_nodes: Number of nodes
        
    Returns:
        edge_index: Updated edge index with self-loops
        edge_weight: Normalized edge weights
    """
    device = edge_index.device
    
    # Add self-loops
    loop_index = torch.arange(num_nodes, device=device).unsqueeze(0).repeat(2, 1)
    edge_index = torch.cat([edge_index, loop_index], dim=1)
    
    # Compute degree
    src, dst = edge_index[0], edge_index[1]
    deg = torch.zeros(num_nodes, device=device)
    deg.scatter_add_(0, dst, torch.ones_like(dst, dtype=torch.float))
    
    # Symmetric normalization
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    
    edge_weight = deg_inv_sqrt[src] * deg_inv_sqrt[dst]
    
    return edge_index, edge_weight


def k_hop_subgraph(
    node_idx: int,
    num_hops: int,
    edge_index: torch.Tensor,
    num_nodes: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract k-hop subgraph around a node.
    
    Args:
        node_idx: Center node index
        num_hops: Number of hops
        edge_index: Edge connectivity [2, num_edges]
        num_nodes: Total number of nodes
        
    Returns:
        subset: Node indices in subgraph
        edge_mask: Mask for edges in subgraph
    """
    subset = {node_idx}
    
    for _ in range(num_hops):
        # Find all neighbors of current subset
        mask = torch.zeros(edge_index.size(1), dtype=torch.bool)
        for node in subset:
            mask |= (edge_index[0] == node)
        
        # Add destination nodes to subset
        new_nodes = edge_index[1, mask].unique().tolist()
        subset.update(new_nodes)
    
    subset = torch.tensor(list(subset), dtype=torch.long)
    
    # Create edge mask
    edge_mask = torch.zeros(edge_index.size(1), dtype=torch.bool)
    for node in subset:
        edge_mask |= ((edge_index[0] == node) & torch.isin(edge_index[1], subset))
    
    return subset, edge_mask


def to_undirected(edge_index: torch.Tensor) -> torch.Tensor:
    """
    Convert directed edges to undirected.
    
    Args:
        edge_index: Directed edge index [2, num_edges]
        
    Returns:
        Undirected edge index
    """
    # Add reverse edges
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    
    # Remove duplicates
    edge_index = torch.unique(edge_index, dim=1)
    
    return edge_index