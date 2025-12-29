"""
Tests for Triton kernels.
"""

import torch
import pytest
from tritongnn.kernels.scatter import scatter_add, scatter_mean, scatter_max


def test_scatter_add():
    """Test scatter_add kernel."""
    src = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    index = torch.tensor([0, 1, 0])
    
    out = scatter_add(src, index, dim_size=2)
    
    # Expected: [1+5, 2+6] for index 0, [3, 4] for index 1
    expected = torch.tensor([[6.0, 8.0], [3.0, 4.0]])
    
    assert torch.allclose(out, expected), f"Got {out}, expected {expected}"
    print("✓ scatter_add test passed")


def test_scatter_mean():
    """Test scatter_mean kernel."""
    src = torch.tensor([[2.0, 4.0], [6.0, 8.0], [10.0, 12.0]])
    index = torch.tensor([0, 1, 0])
    
    out = scatter_mean(src, index, dim_size=2)
    
    # Expected: (2+10)/2, (4+12)/2 for index 0, [6, 8] for index 1
    expected = torch.tensor([[6.0, 8.0], [6.0, 8.0]])
    
    assert torch.allclose(out, expected), f"Got {out}, expected {expected}"
    print("✓ scatter_mean test passed")


def test_scatter_max():
    """Test scatter_max kernel."""
    src = torch.tensor([[1.0, 5.0], [3.0, 2.0], [2.0, 7.0]])
    index = torch.tensor([0, 1, 0])
    
    out = scatter_max(src, index, dim_size=2)
    
    # Expected: max(1, 2), max(5, 7) for index 0, [3, 2] for index 1
    expected = torch.tensor([[2.0, 7.0], [3.0, 2.0]])
    
    assert torch.allclose(out, expected), f"Got {out}, expected {expected}"
    print("✓ scatter_max test passed")


def test_scatter_with_cuda():
    """Test scatter operations on CUDA if available."""
    if not torch.cuda.is_available():
        print("⊘ CUDA not available, skipping CUDA test")
        return
    
    device = torch.device('cuda')
    src = torch.randn(100, 32, device=device)
    index = torch.randint(0, 10, (100,), device=device)
    
    # Test all operations
    out_add = scatter_add(src, index, dim_size=10)
    out_mean = scatter_mean(src, index, dim_size=10)
    out_max = scatter_max(src, index, dim_size=10)
    
    assert out_add.shape == (10, 32)
    assert out_mean.shape == (10, 32)
    assert out_max.shape == (10, 32)
    
    print("✓ CUDA scatter tests passed")


if __name__ == '__main__':
    test_scatter_add()
    test_scatter_mean()
    test_scatter_max()
    test_scatter_with_cuda()
    
    print("\n✓ All kernel tests passed!")