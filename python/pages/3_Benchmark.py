"""
Page 3 - Benchmark Dashboard
==============================
- Live GFLOPS chart across user-selected matrix sizes
- Compares CPU NumPy, Naive GPU, Tiled GPU (16x16 & 32x32), and cuBLAS
- Computes speedup multiples
- GFLOPS = 2*M*K*N / (kernel_time_s * 1e9)
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
build_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'build')
if os.path.exists(build_dir):
    sys.path.insert(0, build_dir)

from mini_cublas import MiniCuBLAS

st.set_page_config(page_title="Benchmark Dashboard", page_icon="", layout="wide")

st.markdown("# Benchmark Dashboard")
st.markdown("**Live GFLOPS comparison across implementations**")
st.markdown("---")

@st.cache_resource
def get_cublas():
    return MiniCuBLAS()

mcb = get_cublas()

# --- Controls ---
col1, col2, col3 = st.columns(3)
with col1:
    operation = st.selectbox("Operation to Benchmark", [
        "Matrix Multiplication (SGEMM)",
        "Transpose",
        "Dot Product",
        "Element-wise Add"
    ])
with col2:
    sizes = st.multiselect(
        "Sizes to Benchmark (N)",
        [128, 256, 512, 1024, 2048, 4096, 8192],
        default=[256, 512, 1024, 2048],
        help="Matrix size (NxN) or Vector length (N)"
    )
with col3:
    num_iters = st.slider("Iterations per size", 5, 50, 20,
                          help="More iterations = more stable timing")

# --- Run Benchmark ---
if st.button("Run Benchmark", type="primary", use_container_width=True):
    if not sizes:
        st.warning("Please select at least one size")
    else:
        results = []
        progress = st.progress(0)
        status = st.empty()

        for idx, size in enumerate(sorted(sizes)):
            status.text(f"Benchmarking size {size}...")
            
            if operation == "Matrix Multiplication (SGEMM)":
                bench = mcb.benchmark(size, size, size, num_iters)
                bench["Size"] = size
                results.append(bench)
            else:
                # Custom Python-driven benchmarks for other ops
                bench = {"Size": size}
                
                # Setup data
                A = np.random.randn(size, size).astype(np.float32)
                if operation == "Element-wise Add":
                    B = np.random.randn(size, size).astype(np.float32)
                elif operation == "Dot Product":
                    A_vec = np.random.randn(size).astype(np.float32)
                    B_vec = np.random.randn(size).astype(np.float32)
                
                # CPU Timing
                cpu_times = []
                for _ in range(min(num_iters, 5)):
                    t0 = time.perf_counter()
                    if operation == "Transpose":
                        _ = A.T.copy()
                    elif operation == "Element-wise Add":
                        _ = A + B
                    elif operation == "Dot Product":
                        _ = np.dot(A_vec, B_vec)
                    cpu_times.append((time.perf_counter() - t0) * 1000)
                bench["cpu_numpy_ms"] = np.mean(cpu_times)
                
                # GPU Timing
                if mcb.gpu_available:
                    gpu_times = []
                    for _ in range(num_iters):
                        if operation == "Transpose":
                            _, meta = mcb.transpose_gpu(A)
                        elif operation == "Element-wise Add":
                            _, meta = mcb.elementwise_add_gpu(A, B)
                        elif operation == "Dot Product":
                            _, meta = mcb.dot_product_gpu(A_vec, B_vec)
                        gpu_times.append(meta["kernel_time_ms"])
                    bench["gpu_ms"] = np.mean(gpu_times)
                else:
                    bench["gpu_ms"] = bench["cpu_numpy_ms"]
                
                results.append(bench)

            progress.progress((idx + 1) / len(sizes))

        status.text("Benchmark complete!")
        progress.empty()

        st.session_state['bench_results'] = results

st.caption("Mini cuBLAS - Benchmark Dashboard - Abdul Wasay")
