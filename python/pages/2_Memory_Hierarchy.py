"""
Page 2 - Memory Hierarchy Explorer
====================================
- Interactive diagram of GPU memory hierarchy:
  Registers -> Shared Memory -> L1/L2 Cache -> Global Memory (DRAM)
- Heatmap comparison of naive vs tiled access patterns per tile
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch

st.set_page_config(page_title="Memory Hierarchy", page_icon="", layout="wide")

st.markdown("# GPU Memory Hierarchy Explorer")
st.markdown("**Understanding the memory wall and why tiling works**")
st.markdown("---")

# ============================================================================
# Memory Hierarchy Diagram
# ============================================================================
st.markdown("### GPU Memory Architecture")

fig_hier, ax = plt.subplots(figsize=(12, 7))
ax.set_xlim(0, 12)
ax.set_ylim(0, 8)
ax.set_aspect('equal')
ax.axis('off')
fig_hier.patch.set_facecolor('#0e1117')

# Hierarchy levels (bottom = slowest/largest, top = fastest/smallest)
levels = [
    {"y": 0.5, "h": 1.2, "label": "Global Memory (DRAM)", "size": "12 GB",
     "bw": "~288 GB/s", "latency": "~400-600 cycles", "color": "#e74c3c"},
    {"y": 2.2, "h": 1.2, "label": "L2 Cache", "size": "3 MB",
     "bw": "~1.5 TB/s", "latency": "~200 cycles", "color": "#e67e22"},
    {"y": 3.9, "h": 1.2, "label": "L1 Cache / Shared Memory", "size": "48-100 KB/SM",
     "bw": "~12 TB/s", "latency": "~20-30 cycles", "color": "#2ecc71"},
    {"y": 5.6, "h": 1.2, "label": "Registers", "size": "256 KB/SM",
     "bw": "~48 TB/s", "latency": "~1 cycle", "color": "#3498db"},
]

for lvl in levels:
    width = 10 - lvl["y"] * 0.8  # narrower at top
    x = (12 - width) / 2
    rect = FancyBboxPatch((x, lvl["y"]), width, lvl["h"],
                          boxstyle="round,pad=0.1",
                          facecolor=lvl["color"], alpha=0.25,
                          edgecolor=lvl["color"], linewidth=2)
    ax.add_patch(rect)
    ax.text(6, lvl["y"] + lvl["h"] / 2 + 0.15, lvl["label"],
            ha='center', va='center', fontsize=13, fontweight='bold',
            color=lvl["color"])
    ax.text(6, lvl["y"] + lvl["h"] / 2 - 0.25,
            f'{lvl["size"]}  |  {lvl["bw"]}  |  {lvl["latency"]}',
            ha='center', va='center', fontsize=9, color='#cccccc')

# Arrows
for i in range(len(levels) - 1):
    y_start = levels[i]["y"] + levels[i]["h"]
    y_end = levels[i + 1]["y"]
    ax.annotate('', xy=(6, y_end), xytext=(6, y_start),
                arrowprops=dict(arrowstyle='->', color='white', lw=2))

ax.text(6, 7.3, "FASTER / SMALLER ->", ha='center', fontsize=11,
        color='#3498db', fontweight='bold')
ax.text(6, 0.1, "SLOWER / LARGER ->", ha='center', fontsize=11,
        color='#e74c3c', fontweight='bold')

st.pyplot(fig_hier)
plt.close(fig_hier)

# ============================================================================
# Access Pattern Heatmaps
# ============================================================================
st.markdown("---")
st.markdown("### Memory Access Pattern Comparison")

mat_size = st.selectbox("Matrix Size", [8, 16, 32], index=0)
tile_size = st.selectbox("Tile Size", [2, 4, 8], index=1)

col_naive, col_tiled = st.columns(2)

# --- Naive Access Pattern ---
# In naive SGEMM, thread (row, col) accesses ALL K elements of row of A and col of B
# -> every element of A and B is accessed N times
naive_access_A = np.ones((mat_size, mat_size)) * mat_size  # each element accessed N times
naive_access_B = np.ones((mat_size, mat_size)) * mat_size

with col_naive:
    st.markdown("#### Naive SGEMM - Global Memory Access")
    fig_n, (ax_na, ax_nb) = plt.subplots(1, 2, figsize=(8, 4))
    fig_n.patch.set_facecolor('#0e1117')

    im_a = ax_na.imshow(naive_access_A, cmap='Reds', vmin=0, vmax=mat_size)
    ax_na.set_title("A Access Count", color='white', fontsize=10)
    ax_na.tick_params(colors='white')

    im_b = ax_nb.imshow(naive_access_B, cmap='Reds', vmin=0, vmax=mat_size)
    ax_nb.set_title("B Access Count", color='white', fontsize=10)
    ax_nb.tick_params(colors='white')

    plt.colorbar(im_b, ax=ax_nb, shrink=0.8)
    st.pyplot(fig_n)
    plt.close(fig_n)

    total_naive = 2 * mat_size ** 3
    st.metric("Total Global Memory Reads", f"{total_naive:,}")
    st.error(f"Every element is accessed **{mat_size}x** - massive bandwidth waste!")

# --- Tiled Access Pattern ---
# In tiled SGEMM, each element is loaded ceil(K/T) times (once per tile phase)
num_phases = mat_size // tile_size
tiled_access_A = np.ones((mat_size, mat_size)) * num_phases
tiled_access_B = np.ones((mat_size, mat_size)) * num_phases

with col_tiled:
    st.markdown("#### Tiled SGEMM - Shared Memory Access")
    fig_t, (ax_ta, ax_tb) = plt.subplots(1, 2, figsize=(8, 4))
    fig_t.patch.set_facecolor('#0e1117')

    im_ta = ax_ta.imshow(tiled_access_A, cmap='Greens', vmin=0, vmax=mat_size)
    ax_ta.set_title("A Access Count", color='white', fontsize=10)
    ax_ta.tick_params(colors='white')

    # Draw tile boundaries
    for i in range(0, mat_size, tile_size):
        ax_ta.axhline(y=i - 0.5, color='white', linewidth=0.5, alpha=0.5)
        ax_ta.axvline(x=i - 0.5, color='white', linewidth=0.5, alpha=0.5)

    im_tb = ax_tb.imshow(tiled_access_B, cmap='Greens', vmin=0, vmax=mat_size)
    ax_tb.set_title("B Access Count", color='white', fontsize=10)
    ax_tb.tick_params(colors='white')

    for i in range(0, mat_size, tile_size):
        ax_tb.axhline(y=i - 0.5, color='white', linewidth=0.5, alpha=0.5)
        ax_tb.axvline(x=i - 0.5, color='white', linewidth=0.5, alpha=0.5)

    plt.colorbar(im_tb, ax=ax_tb, shrink=0.8)
    st.pyplot(fig_t)
    plt.close(fig_t)

    total_tiled = 2 * mat_size ** 3 // tile_size
    st.metric("Total Global Memory Reads", f"{total_tiled:,}")
    st.success(f"Each element accessed only **{num_phases}x** - "
               f"**{tile_size}x reduction!**")

# ============================================================================
# Key Concepts
# ============================================================================
st.markdown("---")
st.markdown("""
### Key Concepts

| Concept | Description |
|---------|-------------|
| **Memory Coalescing** | Consecutive threads access consecutive memory addresses -> single 128-byte transaction per warp |
| **Shared Memory** | Fast on-chip SRAM (~20 cycles latency). Tiles of A and B are loaded here for reuse |
| **Bank Conflicts** | Shared memory has 32 banks. Column-wise access causes conflicts. **Fix: +1 padding** |
| **`__syncthreads()`** | Barrier ensuring all threads in a block have finished loading before computation |
| **Warp Shuffle** | `__shfl_down_sync()` - register-level communication within a warp (0 latency vs shared memory) |

### Memory Traffic Formula

| Method | Global Memory Reads | Formula |
|--------|-------------------|---------|
| **Naive** | 2 x M x N x K | Every element loaded from DRAM for each output |
| **Tiled** | 2 x M x N x K / T | Tile loaded once into shared memory, reused T times |
| **Reduction** | **Tx** fewer reads | Where T = tile size (16 or 32) |
""")

st.caption("Mini cuBLAS - Memory Hierarchy Explorer - Abdul Wasay")
