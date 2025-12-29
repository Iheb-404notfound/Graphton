import torch
import triton
import triton.language as tl

@triton.jit
def scatter_add_kernel(
    src_ptr,
    index_ptr,
    out_ptr,
    num_items,
    feat_dim,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for scatter_add operation.
    Equivalent to: out[index[i]] += src[i]
    """
    pid = tl.program_id(0)
    
    # Calculate which item we're processing
    item_idx = pid
    
    if item_idx >= num_items:
        return
    
    # Load the index for this item
    idx = tl.load(index_ptr + item_idx)
    
    # Process all features for this item
    for f in range(feat_dim):
        src_val = tl.load(src_ptr + item_idx * feat_dim + f)
        out_offset = idx * feat_dim + f
        
        # Atomic add to handle concurrent writes
        tl.atomic_add(out_ptr + out_offset, src_val)


def scatter_add(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """
    Scatter add operation using Triton.
    
    Args:
        src: Source tensor [num_items, feat_dim]
        index: Index tensor [num_items]
        dim_size: Size of output dimension 0
        
    Returns:
        Output tensor [dim_size, feat_dim]
    """
    num_items = src.size(0)
    feat_dim = src.size(1)
    
    # Initialize output
    out = torch.zeros(dim_size, feat_dim, device=src.device, dtype=src.dtype)
    
    # Define grid
    grid = lambda meta: (num_items,)
    
    # Launch kernel
    scatter_add_kernel[grid](
        src, index, out,
        num_items, feat_dim,
        BLOCK_SIZE=1024,
    )
    
    return out


@triton.jit
def scatter_mean_kernel(
    src_ptr,
    index_ptr,
    out_ptr,
    count_ptr,
    num_items,
    feat_dim,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for scatter_mean operation.
    """
    pid = tl.program_id(0)
    item_idx = pid
    
    if item_idx >= num_items:
        return
    
    idx = tl.load(index_ptr + item_idx)
    
    # Increment count atomically
    tl.atomic_add(count_ptr + idx, 1.0)
    
    # Add values
    for f in range(feat_dim):
        src_val = tl.load(src_ptr + item_idx * feat_dim + f)
        out_offset = idx * feat_dim + f
        tl.atomic_add(out_ptr + out_offset, src_val)


def scatter_mean(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """
    Scatter mean operation using Triton.
    
    Args:
        src: Source tensor [num_items, feat_dim]
        index: Index tensor [num_items]
        dim_size: Size of output dimension 0
        
    Returns:
        Output tensor [dim_size, feat_dim]
    """
    num_items = src.size(0)
    feat_dim = src.size(1)
    
    # Initialize output and count
    out = torch.zeros(dim_size, feat_dim, device=src.device, dtype=src.dtype)
    count = torch.zeros(dim_size, device=src.device, dtype=src.dtype)
    
    # Define grid
    grid = lambda meta: (num_items,)
    
    # Launch kernel
    scatter_mean_kernel[grid](
        src, index, out, count,
        num_items, feat_dim,
        BLOCK_SIZE=1024,
    )
    
    # Divide by count to get mean
    count = count.clamp(min=1.0).unsqueeze(1)
    out = out / count
    
    return out


@triton.jit
def scatter_max_kernel(
    src_ptr,
    index_ptr,
    out_ptr,
    num_items,
    feat_dim,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for scatter_max operation.
    """
    pid = tl.program_id(0)
    item_idx = pid
    
    if item_idx >= num_items:
        return
    
    idx = tl.load(index_ptr + item_idx)
    
    for f in range(feat_dim):
        src_val = tl.load(src_ptr + item_idx * feat_dim + f)
        out_offset = idx * feat_dim + f
        tl.atomic_max(out_ptr + out_offset, src_val)


def scatter_max(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """
    Scatter max operation using Triton.
    
    Args:
        src: Source tensor [num_items, feat_dim]
        index: Index tensor [num_items]
        dim_size: Size of output dimension 0
        
    Returns:
        Output tensor [dim_size, feat_dim]
    """
    num_items = src.size(0)
    feat_dim = src.size(1)
    
    # Initialize output with very negative values
    out = torch.full(
        (dim_size, feat_dim),
        float('-inf'),
        device=src.device,
        dtype=src.dtype
    )
    
    # Define grid
    grid = lambda meta: (num_items,)
    
    # Launch kernel
    scatter_max_kernel[grid](
        src, index, out,
        num_items, feat_dim,
        BLOCK_SIZE=1024,
    )
    
    # Replace -inf with 0 for empty bins
    out = torch.where(torch.isinf(out), torch.zeros_like(out), out)
    
    return out