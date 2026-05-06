"""
Page 1 - Tiling Simulator
==========================
Step-through animation of the tiled matrix multiplication algorithm.
- Renders matrices A, B, C as colour-coded grids
- Highlights the active tile currently in shared memory
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

st.set_page_config(page_title="Tiling Simulator", page_icon="", layout="wide")

st.markdown("# Tiling Simulator")
st.markdown("**Step-through animation of the tiled SGEMM algorithm**")
st.markdown("---")

# --- Controls ---
col1, col2, col3 = st.columns(3)
with col1:
    mat_size = st.selectbox("Matrix Size", [4, 8, 16], index=0,
                            help="Size of square matrices A, B, C")
with col2:
    tile_size = st.selectbox("Tile Size", [2, 4], index=0,
                             help="Size of the shared memory tile (TxT)")
with col3:
    st.markdown("")

# Generate matrices
np.random.seed(42)
A = np.random.randint(0, 10, (mat_size, mat_size)).astype(np.float32)
B = np.random.randint(0, 10, (mat_size, mat_size)).astype(np.float32)
C = A @ B

num_tiles = mat_size // tile_size

st.write(f"Matrix size: {mat_size}x{mat_size}, Tile size: {tile_size}x{tile_size}")
st.write(f"Number of tiles per dimension: {num_tiles}")

st.caption("Mini cuBLAS - Tiling Simulator - Abdul Wasay")
