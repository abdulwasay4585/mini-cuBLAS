"""
Page 1 - Tiling Simulator
==========================
Step-through animation of the tiled matrix multiplication algorithm.
- Renders matrices A, B, C as colour-coded grids
- Highlights the active tile currently in shared memory
- Shows a live counter comparing global memory reads: naive vs tiled
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
C = A @ B  # Ground truth

num_tiles = mat_size // tile_size

# --- Tile step slider ---
total_steps = num_tiles * num_tiles * num_tiles  # (row_tile, col_tile, k_tile)
step = st.slider("Step (tile phase)", 0, total_steps - 1, 0,
                 help="Each step corresponds to one tile being loaded into shared memory")

# Decode step into (output tile row, output tile col, k-tile)
k_tile = step % num_tiles
remaining = step // num_tiles
col_tile = remaining % num_tiles
row_tile = remaining // num_tiles

# --- Memory read counters ---
N = mat_size
T = tile_size
naive_reads = 2 * N * N * N  # 2*M*N*K for naive
# For tiled: up to current step
current_tile_step = step + 1
tiled_reads_so_far = current_tile_step * 2 * T * T  # each tile loads T*T from A and T*T from B
tiled_total = 2 * N * N * N // T  # total tiled reads

col_m1, col_m2, col_m3, col_m4 = st.columns(4)
col_m1.metric("Current Tile", f"({row_tile}, {col_tile}, k={k_tile})")
col_m2.metric("Naive Global Reads", f"{naive_reads:,}")
col_m3.metric("Tiled Reads (so far)", f"{tiled_reads_so_far:,}")
col_m4.metric("Reduction Factor", f"{T}x")

st.markdown("---")

# --- Visualization ---
def draw_matrix_with_tile(matrix, title, highlight_row, highlight_col, tile_sz, cmap='YlOrRd'):
    """Draw a matrix as a color grid with a highlighted tile region."""
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.imshow(matrix, cmap=cmap, aspect='equal')

    # Add cell values
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            ax.text(j, i, f"{val:.0f}" if val == int(val) else f"{val:.1f}",
                    ha='center', va='center', fontsize=8,
                    color='white' if val > matrix.max() * 0.6 else 'black')

    # Highlight tile
    if highlight_row is not None and highlight_col is not None:
        rect = patches.Rectangle(
            (highlight_col * tile_sz - 0.5, highlight_row * tile_sz - 0.5),
            tile_sz, tile_sz,
            linewidth=3, edgecolor='#00ff00', facecolor='none', linestyle='--'
        )
        ax.add_patch(rect)

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_yticks(range(matrix.shape[0]))
    ax.grid(True, alpha=0.3)
    return fig

col_a, col_b, col_c = st.columns(3)

with col_a:
    st.markdown("### Matrix A")
    fig_a = draw_matrix_with_tile(A, f"A - Tile row={row_tile}, k={k_tile}",
                                   row_tile, k_tile, tile_size, 'Blues')
    st.pyplot(fig_a)
    plt.close(fig_a)

with col_b:
    st.markdown("### Matrix B")
    fig_b = draw_matrix_with_tile(B, f"B - Tile k={k_tile}, col={col_tile}",
                                   k_tile, col_tile, tile_size, 'Oranges')
    st.pyplot(fig_b)
    plt.close(fig_b)

with col_c:
    st.markdown("### Result C = A x B")
    fig_c = draw_matrix_with_tile(C, f"C - Output tile ({row_tile}, {col_tile})",
                                   row_tile, col_tile, tile_size, 'Greens')
    st.pyplot(fig_c)
    plt.close(fig_c)

# --- Shared Memory View ---
st.markdown("---")
st.markdown("### Current Shared Memory Contents")

sA = A[row_tile*tile_size:(row_tile+1)*tile_size,
       k_tile*tile_size:(k_tile+1)*tile_size]
sB = B[k_tile*tile_size:(k_tile+1)*tile_size,
       col_tile*tile_size:(col_tile+1)*tile_size]

col_sa, col_sb = st.columns(2)
with col_sa:
    st.markdown(f"**`__shared__ sA[{tile_size}][{tile_size}]`** (from A)")
    fig_sa, ax_sa = plt.subplots(figsize=(3, 3))
    ax_sa.imshow(sA, cmap='Blues', aspect='equal')
    for i in range(sA.shape[0]):
        for j in range(sA.shape[1]):
            ax_sa.text(j, i, f"{sA[i,j]:.0f}", ha='center', va='center', fontsize=12)
    ax_sa.set_title("Shared Memory Tile A", fontsize=10)
    st.pyplot(fig_sa)
    plt.close(fig_sa)

with col_sb:
    st.markdown(f"**`__shared__ sB[{tile_size}][{tile_size}]`** (from B)")
    fig_sb, ax_sb = plt.subplots(figsize=(3, 3))
    ax_sb.imshow(sB, cmap='Oranges', aspect='equal')
    for i in range(sB.shape[0]):
        for j in range(sB.shape[1]):
            ax_sb.text(j, i, f"{sB[i,j]:.0f}", ha='center', va='center', fontsize=12)
    ax_sb.set_title("Shared Memory Tile B", fontsize=10)
    st.pyplot(fig_sb)
    plt.close(fig_sb)

# --- Explanation ---
st.markdown("---")
st.markdown(f"""
### At Step {step}

1. **Thread block** at output position `({row_tile}, {col_tile})` is computing tile `k={k_tile}` of {num_tiles}
2. **Loading into shared memory:**
   - `sA` <- `A[{row_tile*tile_size}:{(row_tile+1)*tile_size}, {k_tile*tile_size}:{(k_tile+1)*tile_size}]`
   - `sB` <- `B[{k_tile*tile_size}:{(k_tile+1)*tile_size}, {col_tile*tile_size}:{(col_tile+1)*tile_size}]`
3. **`__syncthreads()`** - barrier ensures all threads finished loading
4. **Accumulate:** each thread computes `sum += sA[ty][k] * sB[k][tx]` for `k = 0..{tile_size-1}`
5. **`__syncthreads()`** - barrier before next tile overwrites shared memory
""")

st.caption("Mini cuBLAS - Tiling Simulator - Abdul Wasay")
