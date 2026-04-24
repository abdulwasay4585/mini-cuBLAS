#pragma once

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// ============================================================================
// CUDA Error Checking Macro
// ============================================================================
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s:%d — %s\n", __FILE__, __LINE__,        \
              cudaGetErrorString(err));                                        \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

// ============================================================================
// Timing Helper - wraps cudaEvent-based GPU timing
// ============================================================================
struct GpuTimer {
  cudaEvent_t start, stop;

  GpuTimer() {
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
  }

  ~GpuTimer() {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }

  void begin(cudaStream_t stream = 0) {
    CUDA_CHECK(cudaEventRecord(start, stream));
  }

  void end(cudaStream_t stream = 0) {
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));
  }

  // Returns elapsed time in milliseconds
  float elapsed_ms() {
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    return ms;
  }
};

// ============================================================================
// GFLOPS calculation for matrix multiply: FLOPS = 2*M*K*N
// ============================================================================
inline float compute_gflops(int M, int K, int N, float time_ms) {
  double flops = 2.0 * (double)M * (double)K * (double)N;
  double seconds = time_ms / 1000.0;
  return (float)(flops / (seconds * 1e9));
}

// ============================================================================
// Grid/Block dimension helpers
// ============================================================================
inline dim3 make_block_2d(int block_size) {
  return dim3(block_size, block_size);
}

inline dim3 make_grid_2d(int width, int height, int block_size) {
  return dim3((width + block_size - 1) / block_size,
              (height + block_size - 1) / block_size);
}

inline dim3 make_grid_1d(int n, int block_size) {
  return dim3((n + block_size - 1) / block_size);
}
