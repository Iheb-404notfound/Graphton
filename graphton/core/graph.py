import torch
from typing import Optional, Tuple

class Graph:
    """
    Core graph data structure for Graphton.
    
    Args:
        x: Node features [num_nodes, num_features]
        edge_index: Edge connectivity [2, num_edges]
        edge_attr: Edge features [num_edges, edge_features] (optional)
        y: Labels [num_nodes] or [num_graphs] (optional)
        batch: Batch assignment for graph-level tasks (optional)
    """
    
    def __init__(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None
    ):
        self.x = x
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.y = y
        self.batch = batch
        
        self._num_nodes = x.size(0)
        self._num_edges = edge_index.size(1)
        
        # Validate edge_index
        if edge_index.size(0) != 2:
            raise ValueError("edge_index must have shape [2, num_edges]")
        
    @property
    def num_nodes(self) -> int:
        return self._num_nodes
    
    @property
    def num_edges(self) -> int:
        return self._num_edges
    
    @property
    def num_features(self) -> int:
        return self.x.size(1)
    
    def to(self, device: torch.device) -> 'Graph':
        """Move graph to device."""
        self.x = self.x.to(device)
        self.edge_index = self.edge_index.to(device)
        if self.edge_attr is not None:
            self.edge_attr = self.edge_attr.to(device)
        if self.y is not None:
            self.y = self.y.to(device)
        if self.batch is not None:
            self.batch = self.batch.to(device)
        return self
    
    def add_self_loops(self) -> 'Graph':
        """Add self-loops to the graph."""
        num_nodes = self.num_nodes
        device = self.edge_index.device
        
        # Create self-loop edges
        loop_index = torch.arange(num_nodes, device=device).unsqueeze(0).repeat(2, 1)
        
        # Concatenate with existing edges
        self.edge_index = torch.cat([self.edge_index, loop_index], dim=1)
        self._num_edges = self.edge_index.size(1)
        
        # Handle edge attributes if present
        if self.edge_attr is not None:
            loop_attr = torch.zeros(
                num_nodes, 
                self.edge_attr.size(1),
                device=device,
                dtype=self.edge_attr.dtype
            )
            self.edge_attr = torch.cat([self.edge_attr, loop_attr], dim=0)
        
        return self
    
    def degree(self, mode: str = 'in') -> torch.Tensor:
        """
        Compute node degrees.
        
        Args:
            mode: 'in', 'out', or 'both'
        """
        num_nodes = self.num_nodes
        device = self.edge_index.device
        
        if mode == 'in':
            idx = self.edge_index[1]
        elif mode == 'out':
            idx = self.edge_index[0]
        elif mode == 'both':
            idx = torch.cat([self.edge_index[0], self.edge_index[1]], dim=0)
        else:
            raise ValueError(f"Invalid mode: {mode}")
        
        deg = torch.zeros(num_nodes, dtype=torch.long, device=device)
        deg.scatter_add_(0, idx, torch.ones_like(idx))
        
        return deg
    
    def __repr__(self) -> str:
        return (f"Graph(num_nodes={self.num_nodes}, "
                f"num_edges={self.num_edges}, "
                f"num_features={self.num_features})")


def batch_graphs(graphs: list) -> Graph:
    """
    Batch multiple graphs into a single disconnected graph.
    
    Args:
        graphs: List of Graph objects
        
    Returns:
        Batched graph with batch assignment
    """
    x_list = []
    edge_index_list = []
    edge_attr_list = []
    y_list = []
    batch_list = []
    
    node_offset = 0
    
    for i, g in enumerate(graphs):
        x_list.append(g.x)
        
        # Offset edge indices
        edge_index_list.append(g.edge_index + node_offset)
        
        if g.edge_attr is not None:
            edge_attr_list.append(g.edge_attr)
        
        if g.y is not None:
            y_list.append(g.y)
        
        # Create batch assignment
        batch_list.append(torch.full((g.num_nodes,), i, dtype=torch.long))
        
        node_offset += g.num_nodes
    
    # Concatenate all components
    x = torch.cat(x_list, dim=0)
    edge_index = torch.cat(edge_index_list, dim=1)
    edge_attr = torch.cat(edge_attr_list, dim=0) if edge_attr_list else None
    y = torch.cat(y_list, dim=0) if y_list else None
    batch = torch.cat(batch_list, dim=0)
    
    return Graph(x, edge_index, edge_attr, y, batch)