// ============================================================================
// Dot Product Kernel - Parallel Tree Reduction + Warp Shuffle
// ============================================================================
// Two-phase reduction:
//
// Phase 1 - Block-level tree reduction in shared memory:
//   Each thread loads one (or more) elements, accumulates a partial sum.
//   Then a classic tree-based reduction halves active threads each step:
//     stride = blockDim.x / 2
//     while stride > 0:
//       if tid < stride:  sdata[tid] += sdata[tid + stride]
//       __syncthreads()
//       stride >>= 1
//
// Phase 2 - Final warp uses __shfl_down_sync() (warp shuffle):
//   When fewer than 32 elements remain, all are within a single warp.
//   Warp shuffles communicate via registers - NO shared memory needed,
//   NO __syncthreads() needed (warps execute in lockstep on SM 8.6).
//   This is an advanced technique NOT covered in standard CUDA curriculum.
//
//   for offset = warpSize/2; offset > 0; offset >>= 1:
//     val += __shfl_down_sync(0xFFFFFFFF, val, offset)
//
// The block-level partial sums are written to a small output array,
// which is then summed on the host (or with a second kernel for very
// large inputs).
// ============================================================================

#include "../utils.cuh"

#define DOT_BLOCK_SIZE 256

// ---------------------------------------------------------------------------
// Warp-level reduction using shuffle instructions
// ---------------------------------------------------------------------------
__device__ float warp_reduce_sum(float val) {
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    val += __shfl_down_sync(0xFFFFFFFF, val, offset);
  }
  return val;
}

// ---------------------------------------------------------------------------
// Dot product kernel: each block produces one partial sum
// ---------------------------------------------------------------------------
__global__ void dot_product_kernel(const float *__restrict__ a,
                                   const float *__restrict__ b,
                                   float *__restrict__ partial_sums, int N) {
  __shared__ float sdata[DOT_BLOCK_SIZE];

  int tid = threadIdx.x;
  int gid = blockIdx.x * blockDim.x + threadIdx.x;

  // Each thread computes its partial dot product (handles grid-stride)
  float sum = 0.0f;
  for (int i = gid; i < N; i += blockDim.x * gridDim.x) {
    sum += a[i] * b[i];
  }
  sdata[tid] = sum;
  __syncthreads();

  // Tree-based reduction in shared memory (down to 32 elements)
  for (int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
    if (tid < stride) {
      sdata[tid] += sdata[tid + stride];
    }
    __syncthreads();
  }

  // Final warp: combine the last 64 elements (stride=32) then warp shuffle
  if (tid < 32) {
    // Add the upper-half shared memory values (stride=32 step)
    volatile float *vsdata = sdata;
    vsdata[tid] += vsdata[tid + 32];
    // Now use warp shuffle for the final 32 -> 1 reduction
    float val = vsdata[tid];
    val = warp_reduce_sum(val);
    if (tid == 0) {
      partial_sums[blockIdx.x] = val;
    }
  }
}

// ---------------------------------------------------------------------------
// Host launcher - returns (dot product value, kernel time in ms)
// ---------------------------------------------------------------------------
float launch_dot_product(const float *d_a, const float *d_b, float *d_result,
                         int N, float *out_value) {
  int num_blocks = (N + DOT_BLOCK_SIZE - 1) / DOT_BLOCK_SIZE;
  if (num_blocks > 1024)
    num_blocks = 1024; // cap grid size

  float *d_partial;
  CUDA_CHECK(cudaMalloc(&d_partial, num_blocks * sizeof(float)));

  GpuTimer timer;
  timer.begin();
  dot_product_kernel<<<num_blocks, DOT_BLOCK_SIZE>>>(d_a, d_b, d_partial, N);
  timer.end();

  CUDA_CHECK(cudaGetLastError());

  // Copy partial sums to host and finalize reduction
  float *h_partial = new float[num_blocks];
  CUDA_CHECK(cudaMemcpy(h_partial, d_partial, num_blocks * sizeof(float),
                        cudaMemcpyDeviceToHost));

  float total = 0.0f;
  for (int i = 0; i < num_blocks; ++i)
    total += h_partial[i];

  if (out_value)
    *out_value = total;
  if (d_result) {
    CUDA_CHECK(
        cudaMemcpy(d_result, &total, sizeof(float), cudaMemcpyHostToDevice));
  }

  delete[] h_partial;
  CUDA_CHECK(cudaFree(d_partial));

  return timer.elapsed_ms();
}
