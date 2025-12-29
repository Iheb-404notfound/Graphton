import torch
import torch.nn as nn
from typing import Optional
from ..kernels.scatter import scatter_add, scatter_mean, scatter_max
from ..kernels.aggregation import aggregate_neighbors


class MessagePassing(nn.Module):
    """
    Base class for message passing layers.
    
    Implements the general message passing framework:
    1. message: Compute messages from source to target nodes
    2. aggregate: Aggregate messages at target nodes
    3. update: Update node features based on aggregated messages
    """
    
    def __init__(self, aggr: str = 'add'):
        """
        Args:
            aggr: Aggregation method ('add', 'mean', 'max')
        """
        super().__init__()
        
        if aggr not in ['add', 'mean', 'max']:
            raise ValueError(f"Unknown aggregation: {aggr}")
        
        self.aggr = aggr
    
    def forward(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass implementing message passing.
        
        Args:
            x: Node features [num_nodes, in_features]
            edge_index: Edge connectivity [2, num_edges]
            edge_attr: Optional edge features [num_edges, edge_features]
            
        Returns:
            Updated node features [num_nodes, out_features]
        """
        # Extract source and target nodes
        src, dst = edge_index[0], edge_index[1]
        
        # Propagate messages
        out = self.propagate(x, edge_index, edge_attr, src, dst)
        
        return out
    
    def propagate(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor],
        src: torch.Tensor,
        dst: torch.Tensor
    ) -> torch.Tensor:
        """
        Propagate messages through the graph.
        """
        # Gather source features
        x_src = x[src]
        x_dst = x[dst] if x.size(0) == dst.max() + 1 else None
        
        # Compute messages
        msg = self.message(x_src, x_dst, edge_attr)
        
        # Aggregate messages
        out = self.aggregate(msg, dst, x.size(0))
        
        # Update node features
        out = self.update(out, x)
        
        return out
    
    def message(
        self,
        x_src: torch.Tensor,
        x_dst: Optional[torch.Tensor],
        edge_attr: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        Construct messages. Override this in subclasses.
        
        Args:
            x_src: Source node features [num_edges, in_features]
            x_dst: Target node features [num_edges, in_features] (optional)
            edge_attr: Edge features [num_edges, edge_features] (optional)
            
        Returns:
            Messages [num_edges, msg_features]
        """
        return x_src
    
    def aggregate(
        self,
        msg: torch.Tensor,
        index: torch.Tensor,
        dim_size: int
    ) -> torch.Tensor:
        """
        Aggregate messages at target nodes.
        
        Args:
            msg: Messages [num_edges, msg_features]
            index: Target node indices [num_edges]
            dim_size: Number of nodes
            
        Returns:
            Aggregated features [num_nodes, msg_features]
        """
        if self.aggr == 'add':
            return scatter_add(msg, index, dim_size)
        elif self.aggr == 'mean':
            return scatter_mean(msg, index, dim_size)
        elif self.aggr == 'max':
            return scatter_max(msg, index, dim_size)
    
    def update(
        self,
        aggr_out: torch.Tensor,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Update node features. Override this in subclasses.
        
        Args:
            aggr_out: Aggregated messages [num_nodes, msg_features]
            x: Current node features [num_nodes, in_features]
            
        Returns:
            Updated features [num_nodes, out_features]
        """
        return aggr_out


class SimpleConv(MessagePassing):
    """
    Simple graph convolution: aggregates neighbor features.
    """
    
    def __init__(self, aggr: str = 'add'):
        super().__init__(aggr=aggr)
    
    def message(self, x_src, x_dst, edge_attr):
        return x_src


class WeightedConv(MessagePassing):
    """
    Weighted graph convolution using edge attributes.
    """
    
    def __init__(self, aggr: str = 'add'):
        super().__init__(aggr=aggr)
    
    def message(self, x_src, x_dst, edge_attr):
        if edge_attr is None:
            return x_src
        # Weight messages by edge attributes
        return x_src * edge_attr.unsqueeze(-1)