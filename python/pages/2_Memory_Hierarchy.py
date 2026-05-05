"""
Page 2 - Memory Hierarchy Explorer
====================================
- Interactive diagram of GPU memory hierarchy:
  Registers -> Shared Memory -> L1/L2 Cache -> Global Memory (DRAM)
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
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
    width = 10 - lvl["y"] * 0.8
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

st.pyplot(fig_hier)
plt.close(fig_hier)

st.caption("Mini cuBLAS - Memory Hierarchy Explorer - Abdul Wasay")
