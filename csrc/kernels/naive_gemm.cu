// ============================================================================
// Naive SGEMM Kernel - Global Memory Baseline
// ============================================================================
// Thread mapping:
//   row = blockIdx.y * blockDim.y + threadIdx.y
//   col = blockIdx.x * blockDim.x + threadIdx.x
//
// Each thread computes one element C[row][col] by iterating over the entire
// K dimension, loading A and B from global memory every time.
//
// Memory behaviour:
//   - Every multiply-add loads 2 floats from global memory (DRAM)
//   - Total global memory loads: 2 * M * N * K  (no data reuse)
//   - Extremely bandwidth-bound; serves as the "slow" reference
//
// Execution:
//   Grid  = dim3(ceil(N/BLOCK), ceil(M/BLOCK))
//   Block = dim3(BLOCK, BLOCK)           - typically BLOCK = 16
// ============================================================================

#include "../utils.cuh"

#define NAIVE_BLOCK_SIZE 16

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------
__global__ void naive_sgemm_kernel(const float *__restrict__ A,
                                   const float *__restrict__ B,
                                   float *__restrict__ C, int M, int N, int K) {
  // Map this thread to its output element
  int row = blockIdx.y * blockDim.y + threadIdx.y; // row in C (and A)
  int col = blockIdx.x * blockDim.x + threadIdx.x; // col in C (and B)

  if (row < M && col < N) {
    float sum = 0.0f;
    // Dot product of row-th row of A and col-th column of B
    for (int k = 0; k < K; ++k) {
      sum += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
  }
}

// ---------------------------------------------------------------------------
// Host launcher - returns kernel execution time in ms
// ---------------------------------------------------------------------------
float launch_naive_sgemm(const float *d_A, const float *d_B, float *d_C, int M,
                         int N, int K) {
  dim3 block = make_block_2d(NAIVE_BLOCK_SIZE);
  dim3 grid = make_grid_2d(N, M, NAIVE_BLOCK_SIZE);

  GpuTimer timer;
  timer.begin();
  naive_sgemm_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
  timer.end();

  CUDA_CHECK(cudaGetLastError());
  return timer.elapsed_ms();
}
