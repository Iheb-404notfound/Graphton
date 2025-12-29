import torch
import torch.nn as nn
import torch.nn.functional as F
from ..core.message_passing import MessagePassing


class GCNConv(MessagePassing):
    """
    Graph Convolutional Network layer.
    
    Implements the GCN operation from Kipf & Welling (2017):
    X' = D^(-1/2) A D^(-1/2) X W
    
    Args:
        in_features: Number of input features
        out_features: Number of output features
        bias: Whether to use bias (default: True)
        normalize: Whether to apply symmetric normalization (default: True)
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        normalize: bool = True
    ):
        super().__init__(aggr='add')
        
        self.in_features = in_features
        self.out_features = out_features
        self.normalize = normalize
        
        # Linear transformation
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters."""
        nn.init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)
    
    def forward(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Node features [num_nodes, in_features]
            edge_index: Edge connectivity [2, num_edges]
            edge_weight: Optional edge weights [num_edges]
            
        Returns:
            Updated node features [num_nodes, out_features]
        """
        # Apply linear transformation
        x = self.linear(x)
        
        # Normalize if requested
        if self.normalize:
            # Compute degree
            src, dst = edge_index[0], edge_index[1]
            deg = torch.zeros(x.size(0), device=x.device)
            deg.scatter_add_(0, dst, torch.ones_like(dst, dtype=torch.float))
            
            # Symmetric normalization: D^(-1/2)
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            
            # Apply normalization to edge weights
            if edge_weight is None:
                edge_weight = torch.ones(edge_index.size(1), device=x.device)
            
            edge_weight = deg_inv_sqrt[src] * edge_weight * deg_inv_sqrt[dst]
        
        # Message passing
        out = self.propagate(x, edge_index, edge_weight, edge_index[0], edge_index[1])
        
        return out
    
    def message(self, x_src, x_dst, edge_attr):
        """Compute messages weighted by normalized edge weights."""
        if edge_attr is None:
            return x_src
        return edge_attr.unsqueeze(-1) * x_src
    
    def __repr__(self):
        return f'{self.__class__.__name__}({self.in_features}, {self.out_features})'


class GCN(nn.Module):
    """
    Multi-layer Graph Convolutional Network.
    
    Args:
        in_features: Number of input features
        hidden_features: Number of hidden features per layer
        out_features: Number of output features
        num_layers: Number of GCN layers
        dropout: Dropout rate (default: 0.5)
    """
    
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        num_layers: int = 2,
        dropout: float = 0.5
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Create layers
        self.convs = nn.ModuleList()
        
        # First layer
        self.convs.append(GCNConv(in_features, hidden_features))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_features, hidden_features))
        
        # Output layer
        if num_layers > 1:
            self.convs.append(GCNConv(hidden_features, out_features))
        else:
            self.convs[0] = GCNConv(in_features, out_features)
    
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
        
        # Final layer (no activation or dropout)
        x = self.convs[-1](x, edge_index)
        
        return x
    
    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'num_layers={self.num_layers}, '
                f'dropout={self.dropout})')