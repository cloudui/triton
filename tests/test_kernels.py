"""
Correctness tests — run these before benchmarking.

  python -m pytest tests/test_kernels.py -v
"""

import torch
import pytest

import sys
sys.path.insert(0, '.')
from kernels.rmsnorm import rmsnorm_pytorch, rmsnorm_native, rmsnorm_triton
from kernels.swiglu import swiglu_pytorch, swiglu_triton


class TestRMSNorm:
    def setup_method(self):
        torch.manual_seed(42)
        self.x = torch.randn(32, 128, device='cuda', dtype=torch.float16)
        self.weight = torch.randn(128, device='cuda', dtype=torch.float16)

    def test_pytorch_rmsnorm_shape(self):
        out = rmsnorm_pytorch(self.x, self.weight)
        assert out.shape == self.x.shape

    def test_pytorch_rmsnorm_not_identity(self):
        out = rmsnorm_pytorch(self.x, self.weight)
        assert not torch.allclose(out, self.x)

    def test_pytorch_matches_native(self):
        ref = rmsnorm_native(self.x, self.weight)
        out = rmsnorm_pytorch(self.x, self.weight)
        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)

    def test_triton_matches_pytorch(self):
        ref = rmsnorm_pytorch(self.x, self.weight)
        out = rmsnorm_triton(self.x, self.weight)
        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


class TestSwiGLU:
    def setup_method(self):
        torch.manual_seed(42)
        self.x = torch.randn(32, 128, device='cuda', dtype=torch.float16)
        self.gate = torch.randn(32, 128, device='cuda', dtype=torch.float16)

    def test_pytorch_swiglu_shape(self):
        out = swiglu_pytorch(self.x, self.gate)
        assert out.shape == self.x.shape

    def test_triton_matches_pytorch(self):
        ref = swiglu_pytorch(self.x, self.gate)
        out = swiglu_triton(self.x, self.gate)
        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


# Uncomment when fused kernel is ready
# class TestFused:
#     def setup_method(self):
#         torch.manual_seed(42)
#         self.x = torch.randn(32, 128, device='cuda', dtype=torch.float16)
#         self.gate = torch.randn(32, 128, device='cuda', dtype=torch.float16)
#         self.weight = torch.randn(128, device='cuda', dtype=torch.float16)
#
#     def test_fused_matches_separate(self):
#         from kernels.fused_rmsnorm_swiglu import fused_rmsnorm_swiglu_triton
#         normed = rmsnorm_pytorch(self.x, self.weight)
#         ref = swiglu_pytorch(normed, self.gate)
#         out = fused_rmsnorm_swiglu_triton(self.x, self.gate, self.weight)
#         torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)
