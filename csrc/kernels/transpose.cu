// ============================================================================
// Tiled Transpose Kernel - Shared Memory with Bank Conflict Avoidance
// ============================================================================
// Strategy:
//   1. Read a TILExTILE block from the input in row-major order (coalesced)
//   2. Store into shared memory WITH +1 PADDING to avoid bank conflicts
//      -> __shared__ float tile[TILE][TILE + 1]
//      Without padding, column-wise access in shared memory would cause
//      all 32 threads in a warp to hit the same bank (32-way conflict).
//      +1 padding staggers accesses across different banks.
//   3. Write from shared memory to the output in transposed position
//      (also coalesced, because shared memory reorder is free)
//
// Why coalescing matters:
//   Global memory is accessed in 128-byte transactions (32 floats).
//   Consecutive threads must access consecutive addresses.
//   - READ:  threads read row[blockIdx.y][col..col+TILE] -> coalesced
//   - WRITE: threads write row[blockIdx.x][col..col+TILE] -> coalesced
//     (shared memory performs the transpose, so the global write is still
//      to consecutive addresses)
// ============================================================================

#include "../utils.cuh"

#define TRANSPOSE_TILE 32

__global__ void transpose_tiled_kernel(const float *__restrict__ input,
                                       float *__restrict__ output, int rows,
                                       int cols) {
  // +1 padding eliminates bank conflicts on column-wise shared memory access
  __shared__ float tile[TRANSPOSE_TILE][TRANSPOSE_TILE + 1];

  // Input coordinates (coalesced read)
  int x_in = blockIdx.x * TRANSPOSE_TILE + threadIdx.x;
  int y_in = blockIdx.y * TRANSPOSE_TILE + threadIdx.y;

  // Load input tile into shared memory (row-major -> coalesced)
  if (x_in < cols && y_in < rows) {
    tile[threadIdx.y][threadIdx.x] = input[y_in * cols + x_in];
  }

  __syncthreads();

  // Output coordinates (transposed block position)
  int x_out = blockIdx.y * TRANSPOSE_TILE + threadIdx.x;
  int y_out = blockIdx.x * TRANSPOSE_TILE + threadIdx.y;

  // Write transposed tile to output (column-major read from shared -> coalesced
  // write)
  if (x_out < rows && y_out < cols) {
    output[y_out * rows + x_out] = tile[threadIdx.x][threadIdx.y];
  }
}

// ---------------------------------------------------------------------------
// Host launcher
// ---------------------------------------------------------------------------
float launch_transpose(const float *d_input, float *d_output, int rows,
                       int cols) {
  dim3 block(TRANSPOSE_TILE, TRANSPOSE_TILE);
  dim3 grid((cols + TRANSPOSE_TILE - 1) / TRANSPOSE_TILE,
            (rows + TRANSPOSE_TILE - 1) / TRANSPOSE_TILE);

  // Limit threads per block for non-square tiles if needed
  // With TILE=32: 32x32 = 1024 threads (max allowed)

  GpuTimer timer;
  timer.begin();
  transpose_tiled_kernel<<<grid, block>>>(d_input, d_output, rows, cols);
  timer.end();

  CUDA_CHECK(cudaGetLastError());
  return timer.elapsed_ms();
}
