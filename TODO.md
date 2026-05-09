# Mini cuBLAS — Development Notes

## Completed
- [x] Naive SGEMM kernel (global memory baseline)
- [x] Tiled SGEMM (16x16 and 32x32 shared memory tiles)
- [x] Matrix transpose with bank conflict avoidance
- [x] Dot product with tree reduction and warp shuffle
- [x] Element-wise operations (add, scalar mul, ReLU, sigmoid)
- [x] pybind11 Python bindings
- [x] Correctness test suite (NumPy validation)
- [x] Streamlit visualizer (4 interactive pages)
- [x] cuBLAS reference benchmark integration

## Potential Future Work
- [ ] Support for half-precision (FP16) kernels
- [ ] Batched GEMM for small matrices
- [ ] Occupancy analysis and auto-tuning tile sizes
- [ ] Memory pool allocator to reduce cudaMalloc overhead
