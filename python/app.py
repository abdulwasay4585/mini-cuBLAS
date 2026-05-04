"""
Mini cuBLAS - Streamlit Visualizer
===================================
Main entrypoint for the multi-page Streamlit application.
Run with: streamlit run python/app.py
"""

import streamlit as st
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Add build directory to path for the compiled CUDA module
build_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'build')
if os.path.exists(build_dir):
    sys.path.insert(0, build_dir)

from mini_cublas import MiniCuBLAS

# Page configuration
st.set_page_config(
    page_title="Mini cuBLAS - GPU Matrix Operations",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium look
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #888;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #333;
        margin: 0.5rem 0;
    }
    .stMetric label { color: #aaa !important; }
</style>
""", unsafe_allow_html=True)

# Initialize the library
@st.cache_resource
def get_cublas():
    return MiniCuBLAS()

mcb = get_cublas()

# Main page
st.markdown('<div class="main-header">Mini cuBLAS</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">GPU-Accelerated Matrix Operations Library</div>',
            unsafe_allow_html=True)

# Device info
info = mcb.device_info()

col1, col2, col3, col4 = st.columns(4)

if info.get("gpu_available"):
    col1.metric("GPU", info.get("name", "Unknown"))
    col2.metric("Compute Cap", info.get("compute_capability", "N/A"))
    col3.metric("VRAM", f"{info.get('global_memory_mb', 0)} MB")
    col4.metric("SMs", info.get("multiprocessor_count", "N/A"))

    st.success("GPU detected - CUDA kernels are active")
else:
    st.warning("Running in CPU simulation mode - CUDA module not compiled")

st.markdown("---")

st.markdown("""
### About This Project

**Mini cuBLAS** is a from-scratch implementation of GPU-accelerated matrix operations,
demonstrating the performance gains of CUDA shared memory tiling, memory coalescing,
and warp-level primitives.

#### Navigation
Use the sidebar to explore the four interactive pages:

| Page | Description |
|------|-------------|
| **Tiling Simulator** | Step-through animation of tiled SGEMM algorithm |
| **Memory Hierarchy** | GPU memory hierarchy diagram + access pattern heatmaps |
| **Benchmark Dashboard** | Live GFLOPS comparison: CPU vs Naive vs Tiled vs cuBLAS |
| **Operation Explorer** | Interactive matrix operations with kernel metadata |

#### Architecture
```
Python (Streamlit UI)
    | pybind11
C++ Host Layer
    | cudaMemcpy H2D
CUDA Kernels (.cu)
    | GPU Global / Shared / Registers
Results -> cudaMemcpy D2H -> Python
```
""")

st.markdown("---")
st.caption("Mini cuBLAS - Abdul Wasay")
