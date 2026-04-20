#  Mini cuBLAS: GPU-Accelerated Linear Algebra & Interactive Visualizer

![CUDA](https://img.shields.io/badge/CUDA-12.0-green)
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28-FF4B4B)
![License](https://img.shields.io/badge/License-MIT-yellow)

**A foundational HPC project that teaches GPU memory hierarchy through a self-built matrix library and an interactive Streamlit simulator.**

Instead of a static C++ wrapper, this project pairs a high-performance CUDA kernel library (bound via `pybind11`) with a rich, multi-page Streamlit visualizer. You can watch tiled matrix multiplication happen in real-time, inspect memory coalescing patterns, and benchmark your own optimized kernels against cuBLAS.

---

##  Project Philosophy

This project is designed to answer the question: *"What is actually happening inside `torch.matmul`?"*

By building:
1.  **Naive GPU kernels** (massive memory bottleneck)
2.  **Tiled Shared Memory kernels** (the classic CUDA optimization)
3.  **A visual debugger** for memory traffic

...you gain an intuitive, visual understanding of the roofline model and the memory wall—skills critical for ML Engineers and HPC Developers.

---

##  The Visualizer Suite (Powered by Streamlit + pybind11)

The non-GPU portion is a dynamic web application. It interacts directly with your compiled CUDA binaries via **pybind11**, offering four interactive pages:

### 1.  Tiling Simulator
- **Step-through animation** of the tiled `SGEMM` algorithm.
- Renders matrices A, B, and C as color-coded grids.
- **Live Highlighting:** Shows the active tile currently residing in `__shared__` memory.
- **Performance Counter:** Displays a real-time counter comparing **Global Memory Reads (Naive)** vs **Global Memory Reads (Tiled)**. See the 16x reduction happen frame-by-frame.

### 2.  Memory Hierarchy Explorer
- Interactive diagram of GPU architecture: Registers → Shared Memory → L1/L2 Cache → Global Memory (DRAM).
- **Heatmap Comparison:** Visualizes memory access patterns per tile. Green for coalesced access, Red for strided/bank-conflicted access.

### 3.  Benchmark Dashboard
- **Live GFLOPS Chart:** Select matrix sizes (256 → 8192) and watch the performance curve update.
- **Multi-Engine Comparison:** Plots four lines on the same graph:
  -  CPU (NumPy)
  -  Naive GPU
  -  Tiled GPU (Your Implementation)
  -  cuBLAS (NVIDIA's Official)
- **Speedup Metrics:** Auto-calculates the multiple improvement from Naive to Tiled vs. the Peak Hardware Theoretical Limit.

### 4.  Operation Explorer
- **Interactive REPL:** Enter custom small matrices (e.g., 4x4).
- **Supported Ops:** `MatMul`, `Transpose`, `Element-wise Add`, `Dot Product`.
- **Kernel Metadata Inspector:** Displays launch configuration (Grid/Block dims) and estimated memory traffic for the selected operation.

---

## 🛠️ Tech Stack

| Component           | Technology                                                                             |
| ------------------- | -------------------------------------------------------------------------------------- |
| **GPU Kernels**     | CUDA C++ (Tiled GEMM, Coalesced Access, Bank Conflict Avoidance)                        |
| **Python Bindings** | `pybind11` (Zero-copy access to GPU arrays)                                             |
| **Visualizer**      | `Streamlit`, `Plotly`, `Matplotlib` (Grid rendering)                                    |
| **Validation**      | `NumPy`, `PyTorch` (Reference checking)                                                 |
| **Build System**    | `CMake` / `Makefile`                                                                    |

---

##  Installation & Setup

### Prerequisites
- NVIDIA GPU (Compute Capability 6.0+)
- CUDA Toolkit 11.8+
- Python 3.10+
- CMake (≥3.18)

### 1. Clone the Repository
```bash
git clone https://github.com/abdulwasay4585/mini-cublas.git
cd mini-cublas

mini_cublas/
├── csrc/                     # CUDA Source Code
│   ├── kernels/
│   │   ├── naive_gemm.cu     # Memory-bound reference
│   │   ├── tiled_gemm.cu     # Shared memory optimized
│   │   └── elementwise.cu
│   ├── bindings.cpp          # pybind11 module definition
│   └── utils.cuh
├── python/                   # Streamlit App & Wrapper
│   ├── app.py                # Main multi-page Streamlit entrypoint
│   ├── pages/                # Individual visualizer pages
│   │   ├── 1_Tiling_Simulator.py
│   │   ├── 2_Memory_Hierarchy.py
│   │   ├── 3_Benchmark.py
│   │   └── 4_Operation_Explorer.py
│   └── mini_cublas.py        # Pure Python class wrapper
├── test/                     # Unit & Integration Tests
│   └── test_correctness.py
├── CMakeLists.txt            # Build configuration
├── requirements.txt
└── README.md





