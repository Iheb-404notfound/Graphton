import torch
import triton
import triton.language as tl

@triton.jit
def aggregate_neighbors_kernel(
    x_ptr,
    edge_index_ptr,
    out_ptr,
    num_edges,
    feat_dim,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Aggregate neighbor features for message passing.
    For each edge (src, dst), copy x[src] to a buffer indexed by edge.
    """
    pid = tl.program_id(0)
    
    if pid >= num_edges:
        return
    
    # Load source node index
    src_idx = tl.load(edge_index_ptr + pid)
    
    # Copy features from source node to output
    for f in range(feat_dim):
        feat_val = tl.load(x_ptr + src_idx * feat_dim + f)
        tl.store(out_ptr + pid * feat_dim + f, feat_val)


def aggregate_neighbors(
    x: torch.Tensor, 
    edge_index: torch.Tensor
) -> torch.Tensor:
    """
    Gather features from source nodes for all edges.
    
    Args:
        x: Node features [num_nodes, feat_dim]
        edge_index: Edge connectivity [2, num_edges]
        
    Returns:
        Aggregated features [num_edges, feat_dim]
    """
    num_edges = edge_index.size(1)
    feat_dim = x.size(1)
    
    # Allocate output
    out = torch.empty(num_edges, feat_dim, device=x.device, dtype=x.dtype)
    
    # Launch kernel - gather from source nodes
    grid = lambda meta: (triton.cdiv(num_edges, 1024),)
    
    aggregate_neighbors_kernel[grid](
        x,
        edge_index[0],  # Source nodes
        out,
        num_edges,
        feat_dim,
        BLOCK_SIZE=1024,
    )
    
    return out


@triton.jit
def spmm_kernel(
    row_ptr,
    col_idx_ptr,
    values_ptr,
    x_ptr,
    out_ptr,
    num_rows,
    feat_dim,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Sparse matrix-matrix multiplication kernel.
    Computes out = A @ x where A is sparse (CSR format).
    """
    row = tl.program_id(0)
    
    if row >= num_rows:
        return
    
    # Get row boundaries
    row_start = tl.load(row_ptr + row)
    row_end = tl.load(row_ptr + row + 1)
    
    # Process each feature dimension
    for f in range(feat_dim):
        result = 0.0
        
        # Iterate over non-zero elements in this row
        for idx in range(row_start, row_end):
            col = tl.load(col_idx_ptr + idx)
            val = tl.load(values_ptr + idx)
            x_val = tl.load(x_ptr + col * feat_dim + f)
            result += val * x_val
        
        # Store result
        tl.store(out_ptr + row * feat_dim + f, result)


def spmm(
    row_ptr: torch.Tensor,
    col_idx: torch.Tensor,
    values: torch.Tensor,
    x: torch.Tensor
) -> torch.Tensor:
    """
    Sparse matrix-matrix multiplication using Triton.
    
    Args:
        row_ptr: Row pointers for CSR format [num_rows + 1]
        col_idx: Column indices [num_nonzero]
        values: Non-zero values [num_nonzero]
        x: Dense matrix [num_cols, feat_dim]
        
    Returns:
        Result [num_rows, feat_dim]
    """
    num_rows = row_ptr.size(0) - 1
    feat_dim = x.size(1)
    
    out = torch.zeros(num_rows, feat_dim, device=x.device, dtype=x.dtype)
    
    grid = lambda meta: (num_rows,)
    
    spmm_kernel[grid](
        row_ptr,
        col_idx,
        values,
        x,
        out,
        num_rows,
        feat_dim,
        BLOCK_SIZE=1024,
    )
    
    return out


def edge_index_to_csr(
    edge_index: torch.Tensor, 
    num_nodes: int,
    edge_weight: torch.Tensor = None
) -> tuple:
    """
    Convert COO format edge_index to CSR format.
    
    Args:
        edge_index: [2, num_edges]
        num_nodes: Number of nodes
        edge_weight: Optional edge weights [num_edges]
        
    Returns:
        (row_ptr, col_idx, values)
    """
    num_edges = edge_index.size(1)
    device = edge_index.device
    
    if edge_weight is None:
        edge_weight = torch.ones(num_edges, device=device)
    
    # Sort by destination (row) then source (col)
    dst, src = edge_index[1], edge_index[0]
    perm = torch.argsort(dst * num_nodes + src)
    
    dst_sorted = dst[perm]
    src_sorted = src[perm]
    weight_sorted = edge_weight[perm]
    
    # Build row pointer
    row_ptr = torch.zeros(num_nodes + 1, dtype=torch.long, device=device)
    row_ptr[1:] = torch.cumsum(
        torch.bincount(dst_sorted, minlength=num_nodes),
        dim=0
    )
    
    return row_ptr, src_sorted, weight_sorted