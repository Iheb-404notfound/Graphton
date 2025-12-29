import torch
import torch.nn as nn
import torch.nn.functional as F
from ..core.message_passing import MessagePassing


class GATConv(MessagePassing):
    """
    Graph Attention Network layer.
    
    Implements the GAT operation from Veličković et al. (2018).
    Uses attention mechanism to weight neighbor contributions.
    
    Args:
        in_features: Number of input features
        out_features: Number of output features
        heads: Number of attention heads (default: 1)
        concat: Whether to concatenate or average heads (default: True)
        dropout: Dropout rate for attention weights (default: 0.0)
        bias: Whether to use bias (default: True)
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        heads: int = 1,
        concat: bool = True,
        dropout: float = 0.0,
        bias: bool = True
    ):
        super().__init__(aggr='add')
        
        self.in_features = in_features
        self.out_features = out_features
        self.heads = heads
        self.concat = concat
        self.dropout_rate = dropout
        
        # Linear transformation for each head
        self.lin = nn.Linear(in_features, heads * out_features, bias=False)
        
        # Attention parameters
        self.att_src = nn.Parameter(torch.Tensor(1, heads, out_features))
        self.att_dst = nn.Parameter(torch.Tensor(1, heads, out_features))
        
        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(heads * out_features))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters."""
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
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
            Updated node features [num_nodes, heads * out_features] if concat
                                   or [num_nodes, out_features] if not concat
        """
        # Linear transformation
        x = self.lin(x).view(-1, self.heads, self.out_features)
        
        # Compute attention scores
        src, dst = edge_index[0], edge_index[1]
        
        # Attention logits: a^T [Wh_i || Wh_j]
        alpha_src = (x * self.att_src).sum(dim=-1)  # [num_nodes, heads]
        alpha_dst = (x * self.att_dst).sum(dim=-1)  # [num_nodes, heads]
        
        alpha = alpha_src[src] + alpha_dst[dst]  # [num_edges, heads]
        alpha = F.leaky_relu(alpha, 0.2)
        
        # Softmax normalization per destination node
        alpha = self.softmax(alpha, dst, x.size(0))
        
        # Apply dropout to attention weights
        alpha = F.dropout(alpha, p=self.dropout_rate, training=self.training)
        
        # Store for message passing
        self._alpha = alpha
        self._x = x
        
        # Propagate
        out = self.propagate(x, edge_index, None, src, dst)
        
        # Concatenate or average heads
        if self.concat:
            out = out.view(-1, self.heads * self.out_features)
        else:
            out = out.mean(dim=1)
        
        # Add bias
        if self.bias is not None:
            out = out + self.bias
        
        return out
    
    def message(self, x_src, x_dst, edge_attr):
        """Apply attention weights to source features."""
        # x_src: [num_edges, heads, out_features]
        # alpha: [num_edges, heads]
        alpha = self._alpha.unsqueeze(-1)  # [num_edges, heads, 1]
        x_src = self._x[self.current_src]  # Get features with proper shape
        return alpha * x_src
    
    def propagate(self, x, edge_index, edge_attr, src, dst):
        """Custom propagate to handle multi-head attention."""
        self.current_src = src
        num_nodes = x.size(0)
        
        # Gather source features
        x_src = x[src]  # [num_edges, heads, out_features]
        
        # Weight by attention
        msg = self._alpha.unsqueeze(-1) * x_src
        
        # Aggregate per head
        out = torch.zeros(
            num_nodes, self.heads, self.out_features,
            device=x.device, dtype=x.dtype
        )
        
        # Sum messages for each destination node
        for h in range(self.heads):
            out[:, h, :].scatter_add_(
                0,
                dst.unsqueeze(-1).expand(-1, self.out_features),
                msg[:, h, :]
            )
        
        return out
    
    def softmax(self, alpha, index, num_nodes):
        """Compute softmax over attention scores grouped by destination node."""
        # Subtract max for numerical stability
        alpha_max = torch.zeros(num_nodes, self.heads, device=alpha.device)
        alpha_max.scatter_reduce_(
            0, index.unsqueeze(-1).expand(-1, self.heads),
            alpha, reduce='amax', include_self=False
        )
        alpha = alpha - alpha_max[index]
        
        # Compute exp
        alpha = alpha.exp()
        
        # Sum over destination nodes
        alpha_sum = torch.zeros(num_nodes, self.heads, device=alpha.device)
        alpha_sum.scatter_add_(
            0, index.unsqueeze(-1).expand(-1, self.heads), alpha
        )
        
        # Normalize
        alpha = alpha / (alpha_sum[index] + 1e-16)
        
        return alpha
    
    def __repr__(self):
        return (f'{self.__class__.__name__}({self.in_features}, '
                f'{self.out_features}, heads={self.heads})')