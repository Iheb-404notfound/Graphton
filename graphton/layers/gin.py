import torch
import torch.nn as nn
import torch.nn.functional as F
from ..core.message_passing import MessagePassing


class GINConv(MessagePassing):
    """
    Graph Isomorphism Network layer.
    
    Implements the GIN operation from Xu et al. (2019):
    x_i' = MLP((1 + ε) * x_i + sum(x_j for j in N(i)))
    
    Args:
        nn: Neural network (MLP) to apply
        eps: Initial epsilon value (default: 0.0)
        train_eps: Whether epsilon is trainable (default: False)
    """
    
    def __init__(
        self,
        nn: nn.Module,
        eps: float = 0.0,
        train_eps: bool = False
    ):
        super().__init__(aggr='add')
        
        self.nn = nn
        self.initial_eps = eps
        
        if train_eps:
            self.eps = nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters."""
        if hasattr(self.nn, 'reset_parameters'):
            self.nn.reset_parameters()
        if isinstance(self.eps, nn.Parameter):
            self.eps.data.fill_(self.initial_eps)
    
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
        
        # Aggregate neighbors
        x_neigh = self.propagate(x, edge_index, None, src, dst)
        
        # Add self-connection with epsilon weighting
        out = (1 + self.eps) * x + x_neigh
        
        # Apply MLP
        out = self.nn(out)
        
        return out
    
    def message(self, x_src, x_dst, edge_attr):
        """Simply return source features."""
        return x_src
    
    def __repr__(self):
        return f'{self.__class__.__name__}(nn={self.nn})'


class GIN(nn.Module):
    """
    Multi-layer Graph Isomorphism Network.
    
    Args:
        in_features: Number of input features
        hidden_features: Number of hidden features per layer
        out_features: Number of output features
        num_layers: Number of GIN layers
        dropout: Dropout rate (default: 0.5)
        train_eps: Whether epsilon is trainable (default: False)
    """
    
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        num_layers: int = 5,
        dropout: float = 0.5,
        train_eps: bool = False
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Create layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # First layer
        mlp = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, hidden_features)
        )
        self.convs.append(GINConv(mlp, train_eps=train_eps))
        self.batch_norms.append(nn.BatchNorm1d(hidden_features))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            mlp = nn.Sequential(
                nn.Linear(hidden_features, hidden_features),
                nn.ReLU(),
                nn.Linear(hidden_features, hidden_features)
            )
            self.convs.append(GINConv(mlp, train_eps=train_eps))
            self.batch_norms.append(nn.BatchNorm1d(hidden_features))
        
        # Output layer
        if num_layers > 1:
            mlp = nn.Sequential(
                nn.Linear(hidden_features, hidden_features),
                nn.ReLU(),
                nn.Linear(hidden_features, out_features)
            )
            self.convs.append(GINConv(mlp, train_eps=train_eps))
        else:
            # Single layer case
            mlp = nn.Sequential(
                nn.Linear(in_features, hidden_features),
                nn.ReLU(),
                nn.Linear(hidden_features, out_features)
            )
            self.convs[0] = GINConv(mlp, train_eps=train_eps)
    
    def forward(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor,
        batch: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass through all layers.
        
        Args:
            x: Node features [num_nodes, in_features]
            edge_index: Edge connectivity [2, num_edges]
            batch: Batch assignment for graph-level tasks (optional)
            
        Returns:
            Output features [num_nodes, out_features] or [num_graphs, out_features]
        """
        # Store intermediate representations for jumping knowledge
        xs = []
        
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            xs.append(x)
        
        # Final layer (no batch norm, activation, or dropout)
        x = self.convs[-1](x, edge_index)
        xs.append(x)
        
        # If batch is provided, perform graph-level pooling
        if batch is not None:
            # Sum pooling across nodes in each graph
            num_graphs = batch.max().item() + 1
            out = torch.zeros(
                num_graphs, x.size(1),
                device=x.device, dtype=x.dtype
            )
            out.scatter_add_(0, batch.unsqueeze(-1).expand_as(x), x)
            return out
        
        return x
    
    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'num_layers={self.num_layers}, '
                f'dropout={self.dropout})')