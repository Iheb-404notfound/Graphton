import torch
import torch.nn as nn
import torch.nn.functional as F
from ..core.message_passing import MessagePassing


class SAGEConv(MessagePassing):
    """
    GraphSAGE layer.
    
    Implements the GraphSAGE operation from Hamilton et al. (2017).
    Combines neighbor aggregation with the node's own features.
    
    Args:
        in_features: Number of input features
        out_features: Number of output features
        aggr: Aggregation method ('mean', 'max', 'add')
        normalize: Whether to apply L2 normalization (default: True)
        bias: Whether to use bias (default: True)
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        aggr: str = 'mean',
        normalize: bool = True,
        bias: bool = True
    ):
        super().__init__(aggr=aggr)
        
        self.in_features = in_features
        self.out_features = out_features
        self.normalize = normalize
        
        # Linear transformation for aggregated neighbors
        self.lin_neigh = nn.Linear(in_features, out_features, bias=False)
        
        # Linear transformation for root node
        self.lin_root = nn.Linear(in_features, out_features, bias=bias)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters."""
        nn.init.xavier_uniform_(self.lin_neigh.weight)
        nn.init.xavier_uniform_(self.lin_root.weight)
        if self.lin_root.bias is not None:
            nn.init.zeros_(self.lin_root.bias)
    
    def forward(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Node features [num_nodes, in_features]
            edge_index: Edge connectivity [2, num_edges]
            
        Returns:
            Updated node features [num_nodes, out_features]
        """
        src, dst = edge_index[0], edge_index[1]
        
        # Aggregate neighbor features
        x_neigh = self.propagate(x, edge_index, None, src, dst)
        
        # Transform aggregated neighbors
        x_neigh = self.lin_neigh(x_neigh)
        
        # Transform root node features
        x_root = self.lin_root(x)
        
        # Combine
        out = x_root + x_neigh
        
        # L2 normalization
        if self.normalize:
            out = F.normalize(out, p=2, dim=-1)
        
        return out
    
    def message(self, x_src, x_dst, edge_attr):
        """Simply return source features."""
        return x_src
    
    def __repr__(self):
        return (f'{self.__class__.__name__}({self.in_features}, '
                f'{self.out_features}, aggr={self.aggr})')


class GraphSAGE(nn.Module):
    """
    Multi-layer GraphSAGE network.
    
    Args:
        in_features: Number of input features
        hidden_features: Number of hidden features per layer
        out_features: Number of output features
        num_layers: Number of SAGE layers
        dropout: Dropout rate (default: 0.0)
        aggr: Aggregation method (default: 'mean')
    """
    
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        num_layers: int = 2,
        dropout: float = 0.0,
        aggr: str = 'mean'
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Create layers
        self.convs = nn.ModuleList()
        
        # First layer
        self.convs.append(SAGEConv(in_features, hidden_features, aggr=aggr))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_features, hidden_features, aggr=aggr))
        
        # Output layer
        if num_layers > 1:
            self.convs.append(SAGEConv(hidden_features, out_features, aggr=aggr))
        else:
            self.convs[0] = SAGEConv(in_features, out_features, aggr=aggr)
    
    def forward(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through all layers.
        
        Args:
            x: Node features [num_nodes, in_features]
            edge_index: Edge connectivity [2, num_edges]
            
        Returns:
            Output features [num_nodes, out_features]
        """
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Final layer
        x = self.convs[-1](x, edge_index)
        
        return x
    
    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'num_layers={self.num_layers}, '
                f'dropout={self.dropout})')