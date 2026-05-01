// ============================================================================
// Tiled SGEMM Kernel - Shared Memory Optimized
// ============================================================================
// Algorithm (per thread-block):
//   for tile t = 0 .. ceil(K / TILE):
//     1. Cooperatively load a TILExTILE sub-block of A into shared memory (sA)
//     2. Cooperatively load a TILExTILE sub-block of B into shared memory (sB)
//     3. __syncthreads()  - ensure all loads are visible
//     4. Each thread accumulates:  sum += sA[ty][k] * sB[k][tx]
//     5. __syncthreads()  - protect shared memory before next tile overwrite
//   C[row][col] = sum
//
// Memory improvement over naive:
//   Naive loads per element:  K  reads from A  +  K  reads from B  =  2K
//   Tiled loads per element:  K/TILE reads from A  +  K/TILE reads from B
//   -> Reduction factor = TILE  (e.g., 16x fewer global memory transactions)
//
// Boundary handling:
//   Threads that map outside the matrix dimensions load 0.0f, which is the
//   additive identity and does not corrupt the accumulation.
// ============================================================================

#include "../utils.cuh"

// ---------------------------------------------------------------------------
// 16x16 Tiled Kernel
// ---------------------------------------------------------------------------
#define TILE_16 16

__global__ void tiled_sgemm_16_kernel(const float *__restrict__ A,
                                      const float *__restrict__ B,
                                      float *__restrict__ C, int M, int N,
                                      int K) {
  // Shared memory tiles for A and B sub-blocks
  __shared__ float sA[TILE_16][TILE_16];
  __shared__ float sB[TILE_16][TILE_16];

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int row = blockIdx.y * TILE_16 + ty; // global row
  int col = blockIdx.x * TILE_16 + tx; // global col

  float sum = 0.0f;

  // Sweep across tiles along the K dimension
  int numTiles = (K + TILE_16 - 1) / TILE_16;
  for (int t = 0; t < numTiles; ++t) {
    // --- Load tile of A into shared memory ---
    int a_col = t * TILE_16 + tx;
    if (row < M && a_col < K)
      sA[ty][tx] = A[row * K + a_col];
    else
      sA[ty][tx] = 0.0f;

    // --- Load tile of B into shared memory ---
    int b_row = t * TILE_16 + ty;
    if (b_row < K && col < N)
      sB[ty][tx] = B[b_row * N + col];
    else
      sB[ty][tx] = 0.0f;

    // Barrier: all threads must finish loading before compute
    __syncthreads();

// --- Accumulate dot product from shared memory ---
#pragma unroll
    for (int k = 0; k < TILE_16; ++k) {
      sum += sA[ty][k] * sB[k][tx];
    }

    // Barrier: protect shared memory before next tile overwrites it
    __syncthreads();
  }

  // Write result
  if (row < M && col < N) {
    C[row * N + col] = sum;
  }
}

// ---------------------------------------------------------------------------
// 32x32 Tiled Kernel (1024 threads per block - near occupancy limit)
// ---------------------------------------------------------------------------
#define TILE_32 32

__global__ void tiled_sgemm_32_kernel(const float *__restrict__ A,
                                      const float *__restrict__ B,
                                      float *__restrict__ C, int M, int N,
                                      int K) {
  __shared__ float sA[TILE_32][TILE_32];
  __shared__ float sB[TILE_32][TILE_32];

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int row = blockIdx.y * TILE_32 + ty;
  int col = blockIdx.x * TILE_32 + tx;

  float sum = 0.0f;

  int numTiles = (K + TILE_32 - 1) / TILE_32;
  for (int t = 0; t < numTiles; ++t) {
    int a_col = t * TILE_32 + tx;
    if (row < M && a_col < K)
      sA[ty][tx] = A[row * K + a_col];
    else
      sA[ty][tx] = 0.0f;

    int b_row = t * TILE_32 + ty;
    if (b_row < K && col < N)
      sB[ty][tx] = B[b_row * N + col];
    else
      sB[ty][tx] = 0.0f;

    __syncthreads();

#pragma unroll
    for (int k = 0; k < TILE_32; ++k) {
      sum += sA[ty][k] * sB[k][tx];
    }

    __syncthreads();
  }

  if (row < M && col < N) {
    C[row * N + col] = sum;
  }
}

// ---------------------------------------------------------------------------
// Host launchers
// ---------------------------------------------------------------------------
float launch_tiled_sgemm_16(const float *d_A, const float *d_B, float *d_C,
                            int M, int N, int K) {
  dim3 block(TILE_16, TILE_16);
  dim3 grid((N + TILE_16 - 1) / TILE_16, (M + TILE_16 - 1) / TILE_16);

  GpuTimer timer;
  timer.begin();
  tiled_sgemm_16_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
  timer.end();

  CUDA_CHECK(cudaGetLastError());
  return timer.elapsed_ms();
}

float launch_tiled_sgemm_32(const float *d_A, const float *d_B, float *d_C,
                            int M, int N, int K) {
  dim3 block(TILE_32, TILE_32);
  dim3 grid((N + TILE_32 - 1) / TILE_32, (M + TILE_32 - 1) / TILE_32);

  GpuTimer timer;
  timer.begin();
  tiled_sgemm_32_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
  timer.end();

  CUDA_CHECK(cudaGetLastError());
  return timer.elapsed_ms();
}
