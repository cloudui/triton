"""
Correctness tests — run these before benchmarking.

  python -m pytest tests/test_kernels.py -v
"""

import sys

import torch

sys.path.insert(0, ".")
from kernels.attention import attention_native, attention_pytorch, attention_triton
from kernels.rmsnorm import rmsnorm_native, rmsnorm_pytorch, rmsnorm_triton
from kernels.softmax import softmax_native, softmax_pytorch, softmax_triton
from kernels.swiglu import swiglu_native, swiglu_pytorch, swiglu_triton


class TestRMSNorm:
    def setup_method(self):
        torch.manual_seed(42)
        self.x = torch.randn(32, 128, device="cuda", dtype=torch.float16)
        self.weight = torch.randn(128, device="cuda", dtype=torch.float16)

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
        self.x = torch.randn(32, 128, device="cuda", dtype=torch.float16)
        self.gate = torch.randn(32, 128, device="cuda", dtype=torch.float16)

    def test_pytorch_swiglu_shape(self):
        out = swiglu_pytorch(self.x, self.gate)
        assert out.shape == self.x.shape

    def test_pytorch_matches_native(self):
        ref = swiglu_native(self.x, self.gate)
        out = swiglu_pytorch(self.x, self.gate)
        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)

    def test_triton_matches_pytorch(self):
        ref = swiglu_pytorch(self.x, self.gate)
        out = swiglu_triton(self.x, self.gate)
        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


class TestFused:
    def setup_method(self):
        torch.manual_seed(42)
        self.x = torch.randn(32, 128, device="cuda", dtype=torch.float16)
        self.gate = torch.randn(32, 128, device="cuda", dtype=torch.float16)
        self.weight = torch.randn(128, device="cuda", dtype=torch.float16)

    def test_fused_matches_separate(self):
        from kernels.fused_rmsnorm_swiglu import fused_rmsnorm_swiglu_triton

        normed = rmsnorm_pytorch(self.x, self.weight)
        ref = swiglu_pytorch(normed, self.gate)
        out = fused_rmsnorm_swiglu_triton(self.x, self.gate, self.weight)
        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


class TestSoftmax:
    def setup_method(self):
        torch.manual_seed(42)
        self.x = torch.randn(32, 128, device="cuda", dtype=torch.float16)

    def test_pytorch_softmax_shape(self):
        out = softmax_pytorch(self.x)
        assert out.shape == self.x.shape

    def test_pytorch_softmax_sums_to_one(self):
        out = softmax_pytorch(self.x)
        row_sums = out.sum(dim=-1)
        torch.testing.assert_close(
            row_sums,
            torch.ones(32, device="cuda", dtype=torch.float16),
            atol=1e-2,
            rtol=1e-2,
        )

    def test_pytorch_matches_native(self):
        ref = softmax_native(self.x)
        out = softmax_pytorch(self.x)
        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)

    def test_triton_matches_pytorch(self):
        ref = softmax_pytorch(self.x)
        out = softmax_triton(self.x)
        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


class TestAttention:
    def setup_method(self):
        torch.manual_seed(42)
        self.seq_len = 64
        self.d_k = 64
        self.Q = torch.randn(self.seq_len, self.d_k, device="cuda", dtype=torch.float16)
        self.K = torch.randn(self.seq_len, self.d_k, device="cuda", dtype=torch.float16)
        self.V = torch.randn(self.seq_len, self.d_k, device="cuda", dtype=torch.float16)

    def test_pytorch_attention_shape(self):
        out = attention_pytorch(self.Q, self.K, self.V)
        assert out.shape == (self.seq_len, self.d_k)

    def test_pytorch_matches_native(self):
        ref = attention_native(self.Q, self.K, self.V)
        out = attention_pytorch(self.Q, self.K, self.V)
        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)

    def test_triton_matches_pytorch(self):
        ref = attention_pytorch(self.Q, self.K, self.V)
        out = attention_triton(self.Q, self.K, self.V)
        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)
