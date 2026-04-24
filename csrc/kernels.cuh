#pragma once

// ============================================================================
// Kernel Launch Declarations
// All functions return kernel execution time in milliseconds (via cudaEvents)
// ============================================================================

// --- Naive SGEMM (global memory only) ---
float launch_naive_sgemm(const float *d_A, const float *d_B, float *d_C, int M,
                         int N, int K);

// --- Tiled SGEMM (shared memory, 16x16 tiles) ---
float launch_tiled_sgemm_16(const float *d_A, const float *d_B, float *d_C,
                            int M, int N, int K);

// --- Tiled SGEMM (shared memory, 32x32 tiles) ---
float launch_tiled_sgemm_32(const float *d_A, const float *d_B, float *d_C,
                            int M, int N, int K);

// --- Tiled Transpose (shared memory + bank conflict avoidance) ---
float launch_transpose(const float *d_input, float *d_output, int rows,
                       int cols);

// --- Dot Product (tree reduction + warp shuffle) ---
float launch_dot_product(const float *d_a, const float *d_b, float *d_result,
                         int N, float *out_value);

// --- Element-wise operations ---
float launch_elementwise_add(const float *d_A, const float *d_B, float *d_C,
                             int N);
float launch_elementwise_scalar_mul(const float *d_A, float scalar, float *d_C,
                                    int N);
float launch_elementwise_relu(const float *d_A, float *d_C, int N);
float launch_elementwise_sigmoid(const float *d_A, float *d_C, int N);
