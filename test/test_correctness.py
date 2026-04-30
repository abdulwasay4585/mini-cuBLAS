"""
Mini cuBLAS - Correctness Test Suite
======================================
Validates all GPU kernel outputs against NumPy CPU reference.
Tolerance: 1e-3 for matmul (float32 non-associativity), 1e-5 for others.
Tests both power-of-two and non-power-of-two matrix sizes.

Run: pytest test/test_correctness.py -v
"""

import numpy as np
import pytest
import sys
import os

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build'))

from mini_cublas import MiniCuBLAS

mcb = MiniCuBLAS()

# ============================================================================
# Matrix Multiplication Tests
# ============================================================================
class TestMatMul:
    @pytest.mark.parametrize("size", [64, 128, 256, 512, 777, 1024])
    def test_naive_sgemm(self, size):
        A = np.random.randn(size, size).astype(np.float32)
        B = np.random.randn(size, size).astype(np.float32)
        expected = A @ B
        got, meta = mcb.matmul_naive_gpu(A, B)
        max_err = np.max(np.abs(expected - got))
        assert max_err < 1e-3, f"Naive SGEMM {size}x{size}: max_err={max_err}"

    @pytest.mark.parametrize("size", [64, 128, 256, 512, 777, 1024])
    def test_tiled_sgemm_16(self, size):
        A = np.random.randn(size, size).astype(np.float32)
        B = np.random.randn(size, size).astype(np.float32)
        expected = A @ B
        got, meta = mcb.matmul_tiled_gpu(A, B, tile_size=16)
        max_err = np.max(np.abs(expected - got))
        assert max_err < 1e-3, f"Tiled-16 SGEMM {size}x{size}: max_err={max_err}"

    @pytest.mark.parametrize("size", [64, 128, 256, 512, 1024])
    def test_tiled_sgemm_32(self, size):
        A = np.random.randn(size, size).astype(np.float32)
        B = np.random.randn(size, size).astype(np.float32)
        expected = A @ B
        got, meta = mcb.matmul_tiled_gpu(A, B, tile_size=32)
        max_err = np.max(np.abs(expected - got))
        assert max_err < 1e-3, f"Tiled-32 SGEMM {size}x{size}: max_err={max_err}"

    def test_non_square(self):
        A = np.random.randn(128, 256).astype(np.float32)
        B = np.random.randn(256, 64).astype(np.float32)
        expected = A @ B
        got, meta = mcb.matmul_tiled_gpu(A, B, tile_size=16)
        max_err = np.max(np.abs(expected - got))
        assert max_err < 1e-3, f"Non-square matmul: max_err={max_err}"

    @pytest.mark.parametrize("size", [256, 512])
    def test_cublas(self, size):
        A = np.random.randn(size, size).astype(np.float32)
        B = np.random.randn(size, size).astype(np.float32)
        expected = A @ B
        got, meta = mcb.matmul_cublas(A, B)
        max_err = np.max(np.abs(expected - got))
        assert max_err < 1e-3, f"cuBLAS {size}x{size}: max_err={max_err}"

# ============================================================================
# Transpose Tests
# ============================================================================
class TestTranspose:
    @pytest.mark.parametrize("rows,cols", [(64, 64), (128, 256), (777, 333), (1024, 512)])
    def test_transpose(self, rows, cols):
        A = np.random.randn(rows, cols).astype(np.float32)
        expected = A.T.copy()
        got, meta = mcb.transpose_gpu(A)
        max_err = np.max(np.abs(expected - got))
        assert max_err < 1e-5, f"Transpose {rows}x{cols}: max_err={max_err}"

# ============================================================================
# Dot Product Tests
# ============================================================================
class TestDotProduct:
    @pytest.mark.parametrize("size", [256, 1024, 10000, 100000])
    def test_dot_product(self, size):
        a = np.random.randn(size).astype(np.float32)
        b = np.random.randn(size).astype(np.float32)
        expected = float(np.dot(a, b))
        got, meta = mcb.dot_product_gpu(a, b)
        rel_err = abs(expected - got) / (abs(expected) + 1e-8)
        assert rel_err < 1e-3, f"Dot product N={size}: rel_err={rel_err}"

# ============================================================================
# Element-wise Tests
# ============================================================================
class TestElementwise:
    @pytest.mark.parametrize("size", [(64, 64), (256, 128), (1024, 1024)])
    def test_add(self, size):
        A = np.random.randn(*size).astype(np.float32)
        B = np.random.randn(*size).astype(np.float32)
        expected = A + B
        got, meta = mcb.elementwise_add_gpu(A, B)
        max_err = np.max(np.abs(expected - got))
        assert max_err < 1e-5, f"Add {size}: max_err={max_err}"

    @pytest.mark.parametrize("size", [(64, 64), (256, 128), (1024, 1024)])
    def test_scalar_mul(self, size):
        A = np.random.randn(*size).astype(np.float32)
        scalar = 3.14
        expected = A * scalar
        got, meta = mcb.elementwise_scalar_mul_gpu(A, scalar)
        max_err = np.max(np.abs(expected - got))
        assert max_err < 1e-5, f"Scalar mul {size}: max_err={max_err}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
