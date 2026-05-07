"""
Page 3 - Benchmark Dashboard
==============================
- Live GFLOPS chart across user-selected matrix sizes
- Compares CPU NumPy, Naive GPU, Tiled GPU (16x16 & 32x32), and cuBLAS
"""

import streamlit as st
import numpy as np
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

st.info("Benchmark page under construction — run button coming soon")

st.caption("Mini cuBLAS - Benchmark Dashboard - Abdul Wasay")
