"""
Mini cuBLAS - Python Wrapper
============================
Provides a clean Python API for the CUDA kernels.
Falls back to NumPy CPU implementations when the CUDA module is not available
(GPU_AVAILABLE = False), so the Streamlit UI works in simulation mode.
"""

import numpy as np
import time

# Try to import the compiled CUDA module
GPU_AVAILABLE = False
try:
    import mini_cublas_cpp as _cuda
    GPU_AVAILABLE = True
except ImportError:
    _cuda = None
    print("[mini_cublas] CUDA module not found - running in CPU-only mode")


class MiniCuBLAS:
    """High-level interface to Mini cuBLAS operations."""

    def __init__(self):
        self.gpu_available = GPU_AVAILABLE

    # ------------------------------------------------------------------
    # Matrix Multiplication
    # ------------------------------------------------------------------
    def matmul_naive_gpu(self, A: np.ndarray, B: np.ndarray):
        """Naive SGEMM on GPU (global memory only)."""
        if not self.gpu_available:
            return self._cpu_matmul(A, B, method="cpu_naive")
        A = np.ascontiguousarray(A, dtype=np.float32)
        B = np.ascontiguousarray(B, dtype=np.float32)
        result, meta = _cuda.matmul_naive(A, B)
        return np.array(result), dict(meta)

    def matmul_tiled_gpu(self, A: np.ndarray, B: np.ndarray, tile_size=16):
        """Tiled SGEMM on GPU (shared memory)."""
        if not self.gpu_available:
            return self._cpu_matmul(A, B, method=f"cpu_tiled_{tile_size}")
        A = np.ascontiguousarray(A, dtype=np.float32)
        B = np.ascontiguousarray(B, dtype=np.float32)
        result, meta = _cuda.matmul_tiled(A, B, tile_size)
        return np.array(result), dict(meta)

    def matmul_cublas(self, A: np.ndarray, B: np.ndarray):
        """cuBLAS SGEMM reference."""
        if not self.gpu_available:
            return self._cpu_matmul(A, B, method="cpu_cublas_sim")
        A = np.ascontiguousarray(A, dtype=np.float32)
        B = np.ascontiguousarray(B, dtype=np.float32)
        result, meta = _cuda.cublas_matmul(A, B)
        return np.array(result), dict(meta)

    def matmul_cpu(self, A: np.ndarray, B: np.ndarray):
        """CPU (NumPy) matrix multiplication."""
        return self._cpu_matmul(A, B, method="cpu_numpy")

    # ------------------------------------------------------------------
    # Transpose
    # ------------------------------------------------------------------
    def transpose_gpu(self, A: np.ndarray):
        """Tiled transpose on GPU."""
        if not self.gpu_available:
            return self._cpu_transpose(A)
        A = np.ascontiguousarray(A, dtype=np.float32)
        result, meta = _cuda.transpose(A)
        return np.array(result), dict(meta)

    # ------------------------------------------------------------------
    # Dot Product
    # ------------------------------------------------------------------
    def dot_product_gpu(self, a: np.ndarray, b: np.ndarray):
        """Dot product on GPU (tree reduction + warp shuffle)."""
        if not self.gpu_available:
            return self._cpu_dot(a, b)
        a = np.ascontiguousarray(a.flatten(), dtype=np.float32)
        b = np.ascontiguousarray(b.flatten(), dtype=np.float32)
        result, meta = _cuda.dot_product(a, b)
        return float(result), dict(meta)

    # ------------------------------------------------------------------
    # Element-wise Operations
    # ------------------------------------------------------------------
    def elementwise_add_gpu(self, A: np.ndarray, B: np.ndarray):
        """Element-wise add on GPU."""
        if not self.gpu_available:
            t0 = time.perf_counter()
            C = A + B
            dt = (time.perf_counter() - t0) * 1000
            return C, {"kernel_time_ms": dt, "method": "cpu_add"}
        A = np.ascontiguousarray(A, dtype=np.float32)
        B = np.ascontiguousarray(B, dtype=np.float32)
        result, meta = _cuda.elementwise_add(A, B)
        return np.array(result), dict(meta)

    def elementwise_scalar_mul_gpu(self, A: np.ndarray, scalar: float):
        """Element-wise scalar multiply on GPU."""
        if not self.gpu_available:
            t0 = time.perf_counter()
            C = A * scalar
            dt = (time.perf_counter() - t0) * 1000
            return C, {"kernel_time_ms": dt, "method": "cpu_scalar_mul"}
        A = np.ascontiguousarray(A, dtype=np.float32)
        result, meta = _cuda.elementwise_scalar_mul(A, scalar)
        return np.array(result), dict(meta)

    # ------------------------------------------------------------------
    # Benchmarking
    # ------------------------------------------------------------------
    def benchmark(self, M, N, K, num_iters=20):
        """Run full benchmark across all implementations."""
        results = {}

        # CPU NumPy
        A = np.random.randn(M, K).astype(np.float32)
        B = np.random.randn(K, N).astype(np.float32)

        # Warm-up
        _ = A @ B

        cpu_times = []
        for _ in range(min(num_iters, 5)):  # fewer CPU iters (slow)
            t0 = time.perf_counter()
            _ = A @ B
            cpu_times.append((time.perf_counter() - t0) * 1000)
        cpu_ms = np.mean(cpu_times)
        flops = 2.0 * M * K * N
        results["cpu_numpy_ms"] = cpu_ms
        results["cpu_numpy_gflops"] = flops / (cpu_ms / 1000) / 1e9

        # GPU benchmarks
        if self.gpu_available:
            gpu_bench = _cuda.benchmark_matmul(M, N, K, num_iters)
            results.update(dict(gpu_bench))

        results["M"] = M
        results["N"] = N
        results["K"] = K

        return results

    # ------------------------------------------------------------------
    # Device Info
    # ------------------------------------------------------------------
    def device_info(self):
        """Get GPU device properties."""
        if not self.gpu_available:
            return {"name": "CPU Only", "gpu_available": False}
        info = dict(_cuda.get_device_info())
        info["gpu_available"] = True
        return info

    # ------------------------------------------------------------------
    # CPU Fallbacks
    # ------------------------------------------------------------------
    def _cpu_matmul(self, A, B, method="cpu"):
        A = np.asarray(A, dtype=np.float32)
        B = np.asarray(B, dtype=np.float32)
        t0 = time.perf_counter()
        C = A @ B
        dt = (time.perf_counter() - t0) * 1000
        M, K = A.shape
        N = B.shape[1]
        gflops = (2.0 * M * K * N) / (dt / 1000) / 1e9
        meta = {"kernel_time_ms": dt, "gflops": gflops, "method": method}
        return C, meta

    def _cpu_transpose(self, A):
        A = np.asarray(A, dtype=np.float32)
        t0 = time.perf_counter()
        C = A.T.copy()
        dt = (time.perf_counter() - t0) * 1000
        meta = {"kernel_time_ms": dt, "method": "cpu_transpose"}
        return C, meta

    def _cpu_dot(self, a, b):
        a = np.asarray(a, dtype=np.float32).flatten()
        b = np.asarray(b, dtype=np.float32).flatten()
        t0 = time.perf_counter()
        result = float(np.dot(a, b))
        dt = (time.perf_counter() - t0) * 1000
        meta = {"kernel_time_ms": dt, "method": "cpu_dot"}
        return result, meta
