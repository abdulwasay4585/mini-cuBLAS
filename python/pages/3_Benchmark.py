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
        st.session_state['bench_operation'] = operation

# --- Display Results ---
if 'bench_results' in st.session_state:
    results = st.session_state['bench_results']
    operation = st.session_state.get('bench_operation', "Matrix Multiplication (SGEMM)")
    sizes_done = [r['Size'] for r in results]

    if operation == "Matrix Multiplication (SGEMM)":
        # Extract GFLOPS
        cpu_gflops = [r.get('cpu_numpy_gflops', 0) for r in results]
        naive_gflops = [r.get('naive_gflops', 0) for r in results]
        tiled16_gflops = [r.get('tiled16_gflops', 0) for r in results]
        tiled32_gflops = [r.get('tiled32_gflops', 0) for r in results]
        cublas_gflops = [r.get('cublas_gflops', 0) for r in results]

        # --- GFLOPS Chart ---
        st.markdown(f"### GFLOPS Performance Comparison: {operation}")

        fig, ax = plt.subplots(figsize=(12, 6))
        fig.patch.set_facecolor('#0e1117')
        ax.set_facecolor('#1a1a2e')

        x = np.arange(len(sizes_done))
        width = 0.15

        bars_cpu = ax.bar(x - 2*width, cpu_gflops, width, label='CPU (NumPy)',
                          color='#95a5a6', alpha=0.9, edgecolor='white', linewidth=0.5)
        bars_naive = ax.bar(x - width, naive_gflops, width, label='Naive GPU',
                            color='#e74c3c', alpha=0.9, edgecolor='white', linewidth=0.5)
        bars_t16 = ax.bar(x, tiled16_gflops, width, label='Tiled 16x16',
                          color='#3498db', alpha=0.9, edgecolor='white', linewidth=0.5)
        bars_t32 = ax.bar(x + width, tiled32_gflops, width, label='Tiled 32x32',
                          color='#2ecc71', alpha=0.9, edgecolor='white', linewidth=0.5)
        bars_cb = ax.bar(x + 2*width, cublas_gflops, width, label='cuBLAS',
                         color='#f39c12', alpha=0.9, edgecolor='white', linewidth=0.5)

        ax.set_xlabel('Matrix Size (NxN)', fontsize=12, color='white')
        ax.set_ylabel('GFLOPS', fontsize=12, color='white')
        ax.set_title('Matrix Multiplication Performance', fontsize=14,
                     fontweight='bold', color='white')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{s}x{s}' for s in sizes_done], color='white')
        ax.tick_params(colors='white')
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(axis='y', alpha=0.2)

        st.pyplot(fig)

else:
    st.info("Click **Run Benchmark** to start profiling")

st.caption("Mini cuBLAS - Benchmark Dashboard - Abdul Wasay")
