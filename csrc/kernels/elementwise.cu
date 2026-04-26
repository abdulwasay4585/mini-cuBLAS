// ============================================================================
// Element-wise Operations - Add, Scalar Multiply, ReLU, Sigmoid
// ============================================================================
// These are 1-D grid kernels. Each thread processes one element.
//
// Memory behaviour:
//   - Fully memory-bandwidth bound (1 or 2 reads + 1 write per element)
//   - Minimal arithmetic intensity (~1 FLOP per byte transferred)
//   - Demonstrates memory coalescing: consecutive threads access
//     consecutive addresses -> single 128-byte transaction per warp
//
// Grid configuration:
//   Block = 256 threads
//   Grid  = ceil(N / 256)
// ============================================================================

#include "../utils.cuh"

#define EW_BLOCK_SIZE 256

// ---------------------------------------------------------------------------
// Element-wise addition: C[i] = A[i] + B[i]
// ---------------------------------------------------------------------------
__global__ void elementwise_add_kernel(const float *__restrict__ A,
                                       const float *__restrict__ B,
                                       float *__restrict__ C, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    C[idx] = A[idx] + B[idx];
  }
}

// ---------------------------------------------------------------------------
// Scalar multiplication: C[i] = A[i] * scalar
// ---------------------------------------------------------------------------
__global__ void elementwise_scalar_mul_kernel(const float *__restrict__ A,
                                              float scalar,
                                              float *__restrict__ C, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    C[idx] = A[idx] * scalar;
  }
}

// ---------------------------------------------------------------------------
// ReLU activation: C[i] = max(0, A[i])
// ---------------------------------------------------------------------------
__global__ void elementwise_relu_kernel(const float *__restrict__ A,
                                        float *__restrict__ C, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    C[idx] = fmaxf(0.0f, A[idx]);
  }
}

// ---------------------------------------------------------------------------
// Sigmoid activation: C[i] = 1 / (1 + exp(-A[i]))
// ---------------------------------------------------------------------------
__global__ void elementwise_sigmoid_kernel(const float *__restrict__ A,
                                           float *__restrict__ C, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    C[idx] = 1.0f / (1.0f + expf(-A[idx]));
  }
}

// ---------------------------------------------------------------------------
// Host launchers
// ---------------------------------------------------------------------------
float launch_elementwise_add(const float *d_A, const float *d_B, float *d_C,
                             int N) {
  dim3 block(EW_BLOCK_SIZE);
  dim3 grid = make_grid_1d(N, EW_BLOCK_SIZE);

  GpuTimer timer;
  timer.begin();
  elementwise_add_kernel<<<grid, block>>>(d_A, d_B, d_C, N);
  timer.end();
  CUDA_CHECK(cudaGetLastError());
  return timer.elapsed_ms();
}

float launch_elementwise_scalar_mul(const float *d_A, float scalar, float *d_C,
                                    int N) {
  dim3 block(EW_BLOCK_SIZE);
  dim3 grid = make_grid_1d(N, EW_BLOCK_SIZE);

  GpuTimer timer;
  timer.begin();
  elementwise_scalar_mul_kernel<<<grid, block>>>(d_A, scalar, d_C, N);
  timer.end();
  CUDA_CHECK(cudaGetLastError());
  return timer.elapsed_ms();
}

float launch_elementwise_relu(const float *d_A, float *d_C, int N) {
  dim3 block(EW_BLOCK_SIZE);
  dim3 grid = make_grid_1d(N, EW_BLOCK_SIZE);

  GpuTimer timer;
  timer.begin();
  elementwise_relu_kernel<<<grid, block>>>(d_A, d_C, N);
  timer.end();
  CUDA_CHECK(cudaGetLastError());
  return timer.elapsed_ms();
}

float launch_elementwise_sigmoid(const float *d_A, float *d_C, int N) {
  dim3 block(EW_BLOCK_SIZE);
  dim3 grid = make_grid_1d(N, EW_BLOCK_SIZE);

  GpuTimer timer;
  timer.begin();
  elementwise_sigmoid_kernel<<<grid, block>>>(d_A, d_C, N);
  timer.end();
  CUDA_CHECK(cudaGetLastError());
  return timer.elapsed_ms();
}
