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
        plt.close(fig)

        # --- Line Chart ---
        st.markdown("### Performance Scaling")

        fig2, ax2 = plt.subplots(figsize=(12, 5))
        fig2.patch.set_facecolor('#0e1117')
        ax2.set_facecolor('#1a1a2e')

        ax2.plot(sizes_done, cpu_gflops, 'o-', label='CPU (NumPy)', color='#95a5a6',
                 linewidth=2, markersize=8)
        ax2.plot(sizes_done, naive_gflops, 's-', label='Naive GPU', color='#e74c3c',
                 linewidth=2, markersize=8)
        ax2.plot(sizes_done, tiled16_gflops, '^-', label='Tiled 16x16', color='#3498db',
                 linewidth=2, markersize=8)
        ax2.plot(sizes_done, tiled32_gflops, 'D-', label='Tiled 32x32', color='#2ecc71',
                 linewidth=2, markersize=8)
        ax2.plot(sizes_done, cublas_gflops, '*-', label='cuBLAS', color='#f39c12',
                 linewidth=2, markersize=10)

        ax2.set_xlabel('Matrix Size (N)', fontsize=12, color='white')
        ax2.set_ylabel('GFLOPS', fontsize=12, color='white')
        ax2.set_title('GFLOPS vs Matrix Size', fontsize=14, fontweight='bold', color='white')
        ax2.tick_params(colors='white')
        ax2.legend(fontsize=10)
        ax2.grid(alpha=0.2)

        st.pyplot(fig2)
        plt.close(fig2)

        # --- Speedup Table ---
        st.markdown("### Speedup Table")
        st.markdown("*Speedup relative to CPU (NumPy)*")

        table_data = []
        for i, r in enumerate(results):
            cpu = r.get('cpu_numpy_gflops', 1)
            if cpu == 0:
                cpu = 1
            row = {
                "Matrix Size": f"{r['Size']}x{r['Size']}",
                "CPU (GFLOPS)": f"{cpu:.1f}",
                "Naive GPU": f"{r.get('naive_gflops', 0):.1f} ({r.get('naive_gflops', 0)/cpu:.1f}x)",
                "Tiled 16x16": f"{r.get('tiled16_gflops', 0):.1f} ({r.get('tiled16_gflops', 0)/cpu:.1f}x)",
                "Tiled 32x32": f"{r.get('tiled32_gflops', 0):.1f} ({r.get('tiled32_gflops', 0)/cpu:.1f}x)",
                "cuBLAS": f"{r.get('cublas_gflops', 0):.1f} ({r.get('cublas_gflops', 0)/cpu:.1f}x)",
            }
            table_data.append(row)

        st.table(table_data)

        # --- Timing Table ---
        st.markdown("### Execution Time (ms)")
        timing_data = []
        for r in results:
            timing_data.append({
                "Size": f"{r['Size']}x{r['Size']}",
                "CPU (ms)": f"{r.get('cpu_numpy_ms', 0):.2f}",
                "Naive (ms)": f"{r.get('naive_ms', 0):.3f}",
                "Tiled 16 (ms)": f"{r.get('tiled16_ms', 0):.3f}",
                "Tiled 32 (ms)": f"{r.get('tiled32_ms', 0):.3f}",
                "cuBLAS (ms)": f"{r.get('cublas_ms', 0):.3f}",
            })
        st.table(timing_data)

    else:
        # Non-SGEMM metrics (Execution Time instead of GFLOPS)
        cpu_ms = [r.get('cpu_numpy_ms', 0) for r in results]
        gpu_ms = [r.get('gpu_ms', 0) for r in results]

        st.markdown(f"### Execution Time Comparison: {operation}")

        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor('#0e1117')
        ax.set_facecolor('#1a1a2e')

        x = np.arange(len(sizes_done))
        width = 0.35

        ax.bar(x - width/2, cpu_ms, width, label='CPU (NumPy)',
               color='#95a5a6', alpha=0.9, edgecolor='white', linewidth=0.5)
        ax.bar(x + width/2, gpu_ms, width, label='GPU Kernel',
               color='#2ecc71', alpha=0.9, edgecolor='white', linewidth=0.5)

        ax.set_xlabel('Size (N)', fontsize=12, color='white')
        ax.set_ylabel('Execution Time (ms) - Lower is Better', fontsize=12, color='white')
        ax.set_title(f'{operation} Performance', fontsize=14, fontweight='bold', color='white')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{s}' for s in sizes_done], color='white')
        ax.tick_params(colors='white')
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(axis='y', alpha=0.2)

        st.pyplot(fig)
        plt.close(fig)

        st.markdown("### Speedup Table")
        table_data = []
        for r in results:
            cpu_t = r.get('cpu_numpy_ms', 1)
            gpu_t = r.get('gpu_ms', 1)
            if gpu_t == 0: gpu_t = 0.001
            speedup = cpu_t / gpu_t
            
            row = {
                "Size (N)": f"{r['Size']}",
                "CPU Time (ms)": f"{cpu_t:.4f}",
                "GPU Time (ms)": f"{gpu_t:.4f}",
                "Speedup vs CPU": f"{speedup:.1f}x"
            }
            table_data.append(row)

        st.table(table_data)

else:
    st.info("Click **Run Benchmark** to start profiling")

    # Show expected results
    st.markdown("""
    ### Expected Results (RTX A2000 12GB)

    | Implementation | Expected GFLOPS | Speedup vs CPU |
    |---|---|---|
    | CPU NumPy (OpenBLAS) | 50-200 | 1x |
    | Naive GPU | 200-500 | ~3x |
    | Tiled GPU (16x16) | 2,000-5,000 | ~25x |
    | Tiled GPU (32x32) | 4,000-8,000 | ~50x |
    | cuBLAS | 10,000-15,000 | ~100x |

    **GFLOPS Formula:** `GFLOPS = 2 x M x K x N / (t_kernel x 10⁹)`
    """)

st.caption("Mini cuBLAS - Benchmark Dashboard - Abdul Wasay")
