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
    page_icon="\u26a1",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize the library
@st.cache_resource
def get_cublas():
    return MiniCuBLAS()

mcb = get_cublas()

# Main page
st.title("Mini cuBLAS")
st.write("GPU-Accelerated Matrix Operations Library")

# Device info
info = mcb.device_info()

if info.get("gpu_available"):
    st.success("GPU detected - CUDA kernels are active")
    st.write(f"GPU: {info.get('name', 'Unknown')}")
else:
    st.warning("Running in CPU simulation mode - CUDA module not compiled")
