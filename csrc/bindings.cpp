// ============================================================================
// pybind11 Bindings - Expose CUDA kernels to Python
// ============================================================================
// Architecture flow:
//   Python (numpy array) -> pybind11 -> C++ host code -> cudaMemcpy H2D ->
//   kernel launch -> cudaMemcpy D2H -> pybind11 -> Python (numpy array +
//   metadata dict)

// Each function:
//   1. Extracts raw pointer + shape from pybind11::array_t<float>
//   2. Allocates device memory (cudaMalloc)
//   3. Copies input H2D (cudaMemcpyHostToDevice)
//   4. Launches kernel, records timing via cudaEvents
//   5. Copies result D2H (cudaMemcpyDeviceToHost)
//   6. Frees device memory
//   7. Returns (result_array, metadata_dict) to Python
// ============================================================================

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <map>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>

#include "kernels.cuh"
#include "utils.cuh"

namespace py = pybind11;

// ============================================================================
// Helper: allocate device memory and copy from host
// ============================================================================
static float *to_device(const float *host_ptr, size_t count) {
  float *d_ptr;
  CUDA_CHECK(cudaMalloc(&d_ptr, count * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_ptr, host_ptr, count * sizeof(float),
                        cudaMemcpyHostToDevice));
  return d_ptr;
}

static void from_device(float *host_ptr, const float *d_ptr, size_t count) {
  CUDA_CHECK(cudaMemcpy(host_ptr, d_ptr, count * sizeof(float),
                        cudaMemcpyDeviceToHost));
}

// ============================================================================
// Matrix Multiplication - Naive
// ============================================================================
static py::tuple matmul_naive(py::array_t<float> A, py::array_t<float> B) {
  auto a = A.unchecked<2>();
  auto b = B.unchecked<2>();
  int M = a.shape(0), K = a.shape(1);
  int N = b.shape(1);

  // Allocate device memory
  float *d_A = to_device(a.data(0, 0), M * K);
  float *d_B = to_device(b.data(0, 0), K * N);
  float *d_C;
  CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));

  // Launch kernel
  float time_ms = launch_naive_sgemm(d_A, d_B, d_C, M, N, K);

  // Copy result back
  auto C = py::array_t<float>({M, N});
  from_device(C.mutable_data(), d_C, M * N);

  // Cleanup
  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B));
  CUDA_CHECK(cudaFree(d_C));

  // Build metadata
  py::dict meta;
  meta["kernel_time_ms"] = time_ms;
  meta["gflops"] = compute_gflops(M, K, N, time_ms);
  meta["method"] = "naive_gpu";
  meta["grid"] = py::make_tuple((N + 15) / 16, (M + 15) / 16);
  meta["block"] = py::make_tuple(16, 16);
  meta["global_memory_reads"] = 2L * M * N * K;

  return py::make_tuple(C, meta);
}

// ============================================================================
// Matrix Multiplication - Tiled (16x16 or 32x32)
// ============================================================================
static py::tuple matmul_tiled(py::array_t<float> A, py::array_t<float> B,
                              int tile_size = 16) {
  auto a = A.unchecked<2>();
  auto b = B.unchecked<2>();
  int M = a.shape(0), K = a.shape(1);
  int N = b.shape(1);

  float *d_A = to_device(a.data(0, 0), M * K);
  float *d_B = to_device(b.data(0, 0), K * N);
  float *d_C;
  CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));

  float time_ms;
  if (tile_size == 32)
    time_ms = launch_tiled_sgemm_32(d_A, d_B, d_C, M, N, K);
  else
    time_ms = launch_tiled_sgemm_16(d_A, d_B, d_C, M, N, K);

  auto C = py::array_t<float>({M, N});
  from_device(C.mutable_data(), d_C, M * N);

  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B));
  CUDA_CHECK(cudaFree(d_C));

  py::dict meta;
  meta["kernel_time_ms"] = time_ms;
  meta["gflops"] = compute_gflops(M, K, N, time_ms);
  meta["method"] = tile_size == 32 ? "tiled_gpu_32" : "tiled_gpu_16";
  meta["tile_size"] = tile_size;
  meta["grid"] = py::make_tuple((N + tile_size - 1) / tile_size,
                                (M + tile_size - 1) / tile_size);
  meta["block"] = py::make_tuple(tile_size, tile_size);
  long tiled_reads = (2L * M * N * K) / tile_size;
  meta["global_memory_reads"] = tiled_reads;

  return py::make_tuple(C, meta);
}

// ============================================================================
// Transpose
// ============================================================================
static py::tuple transpose_op(py::array_t<float> input) {
  auto inp = input.unchecked<2>();
  int rows = inp.shape(0), cols = inp.shape(1);

  float *d_in = to_device(inp.data(0, 0), rows * cols);
  float *d_out;
  CUDA_CHECK(cudaMalloc(&d_out, rows * cols * sizeof(float)));

  float time_ms = launch_transpose(d_in, d_out, rows, cols);

  auto result = py::array_t<float>({cols, rows});
  from_device(result.mutable_data(), d_out, rows * cols);

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));

  py::dict meta;
  meta["kernel_time_ms"] = time_ms;
  meta["method"] = "tiled_transpose";
  meta["block"] = py::make_tuple(32, 32);
  meta["grid"] = py::make_tuple((cols + 31) / 32, (rows + 31) / 32);
  meta["technique"] = "shared_memory_with_bank_conflict_padding";

  return py::make_tuple(result, meta);
}

// ============================================================================
// Dot Product
// ============================================================================
static py::tuple dot_product(py::array_t<float> a, py::array_t<float> b) {
  auto av = a.unchecked<1>();
  auto bv = b.unchecked<1>();
  int N = av.shape(0);

  float *d_a = to_device(av.data(0), N);
  float *d_b = to_device(bv.data(0), N);

  float result_val = 0.0f;
  float time_ms = launch_dot_product(d_a, d_b, nullptr, N, &result_val);

  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));

  py::dict meta;
  meta["kernel_time_ms"] = time_ms;
  meta["method"] = "tree_reduction_warp_shuffle";
  meta["block"] = 256;
  int nb = (N + 255) / 256;
  if (nb > 1024)
    nb = 1024;
  meta["grid"] = nb;
  meta["technique"] = "shfl_down_sync_final_warp";

  return py::make_tuple(result_val, meta);
}

// ============================================================================
// Element-wise Add
// ============================================================================
static py::tuple ew_add(py::array_t<float> A, py::array_t<float> B) {
  int N = A.size();
  float *d_A = to_device(A.data(), N);
  float *d_B = to_device(B.data(), N);
  float *d_C;
  CUDA_CHECK(cudaMalloc(&d_C, N * sizeof(float)));

  float time_ms = launch_elementwise_add(d_A, d_B, d_C, N);

  auto C = py::array_t<float>(A.request().shape);
  from_device(C.mutable_data(), d_C, N);

  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B));
  CUDA_CHECK(cudaFree(d_C));

  py::dict meta;
  meta["kernel_time_ms"] = time_ms;
  meta["method"] = "elementwise_add";
  meta["block"] = 256;
  meta["grid"] = (N + 255) / 256;

  return py::make_tuple(C, meta);
}

// ============================================================================
// Element-wise Scalar Multiply
// ============================================================================
static py::tuple ew_scalar_mul(py::array_t<float> A, float scalar) {
  int N = A.size();
  float *d_A = to_device(A.data(), N);
  float *d_C;
  CUDA_CHECK(cudaMalloc(&d_C, N * sizeof(float)));

  float time_ms = launch_elementwise_scalar_mul(d_A, scalar, d_C, N);

  auto C = py::array_t<float>(A.request().shape);
  from_device(C.mutable_data(), d_C, N);

  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_C));

  py::dict meta;
  meta["kernel_time_ms"] = time_ms;
  meta["method"] = "elementwise_scalar_mul";
  meta["block"] = 256;
  meta["grid"] = (N + 255) / 256;

  return py::make_tuple(C, meta);
}

// ============================================================================
// cuBLAS SGEMM - Reference Implementation
// ============================================================================
static py::tuple cublas_matmul(py::array_t<float> A, py::array_t<float> B) {
  auto a = A.unchecked<2>();
  auto b = B.unchecked<2>();
  int M = a.shape(0), K = a.shape(1);
  int N = b.shape(1);

  float *d_A = to_device(a.data(0, 0), M * K);
  float *d_B = to_device(b.data(0, 0), K * N);
  float *d_C;
  CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));

  cublasHandle_t handle;
  cublasCreate(&handle);

  float alpha = 1.0f, beta = 0.0f;

  // cuBLAS uses column-major, so we compute C^T = B^T * A^T
  // which gives us C in row-major (our format)
  GpuTimer timer;
  timer.begin();
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_A, K,
              &beta, d_C, N);
  timer.end();
  float time_ms = timer.elapsed_ms();

  auto C = py::array_t<float>({M, N});
  from_device(C.mutable_data(), d_C, M * N);

  cublasDestroy(handle);
  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B));
  CUDA_CHECK(cudaFree(d_C));

  py::dict meta;
  meta["kernel_time_ms"] = time_ms;
  meta["gflops"] = compute_gflops(M, K, N, time_ms);
  meta["method"] = "cublas";

  return py::make_tuple(C, meta);
}

// ============================================================================
// Benchmark helper: run a matmul method N times and return avg time
// ============================================================================
static py::dict benchmark_matmul(int M, int N, int K, int num_iters = 20) {
  // Allocate random data on host
  std::vector<float> h_A(M * K), h_B(K * N);
  srand(42);
  for (int i = 0; i < M * K; ++i)
    h_A[i] = (float)rand() / RAND_MAX;
  for (int i = 0; i < K * N; ++i)
    h_B[i] = (float)rand() / RAND_MAX;

  float *d_A = to_device(h_A.data(), M * K);
  float *d_B = to_device(h_B.data(), K * N);
  float *d_C;
  CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));

  // Warm-up launches
  launch_naive_sgemm(d_A, d_B, d_C, M, N, K);
  launch_tiled_sgemm_16(d_A, d_B, d_C, M, N, K);
  launch_tiled_sgemm_32(d_A, d_B, d_C, M, N, K);

  // Benchmark each method
  float naive_ms = 0, tiled16_ms = 0, tiled32_ms = 0, cublas_ms = 0;

  for (int i = 0; i < num_iters; ++i) {
    naive_ms += launch_naive_sgemm(d_A, d_B, d_C, M, N, K);
    tiled16_ms += launch_tiled_sgemm_16(d_A, d_B, d_C, M, N, K);
    tiled32_ms += launch_tiled_sgemm_32(d_A, d_B, d_C, M, N, K);
  }
  naive_ms /= num_iters;
  tiled16_ms /= num_iters;
  tiled32_ms /= num_iters;

  // cuBLAS benchmark
  cublasHandle_t handle;
  cublasCreate(&handle);
  float alpha = 1.0f, beta = 0.0f;

  // warm-up
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_A, K,
              &beta, d_C, N);
  CUDA_CHECK(cudaDeviceSynchronize());

  for (int i = 0; i < num_iters; ++i) {
    GpuTimer t;
    t.begin();
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_A,
                K, &beta, d_C, N);
    t.end();
    cublas_ms += t.elapsed_ms();
  }
  cublas_ms /= num_iters;

  cublasDestroy(handle);
  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B));
  CUDA_CHECK(cudaFree(d_C));

  py::dict result;
  result["M"] = M;
  result["N"] = N;
  result["K"] = K;
  result["num_iters"] = num_iters;
  result["naive_ms"] = naive_ms;
  result["tiled16_ms"] = tiled16_ms;
  result["tiled32_ms"] = tiled32_ms;
  result["cublas_ms"] = cublas_ms;
  result["naive_gflops"] = compute_gflops(M, K, N, naive_ms);
  result["tiled16_gflops"] = compute_gflops(M, K, N, tiled16_ms);
  result["tiled32_gflops"] = compute_gflops(M, K, N, tiled32_ms);
  result["cublas_gflops"] = compute_gflops(M, K, N, cublas_ms);

  return result;
}

// ============================================================================
// Device Info
// ============================================================================
static py::dict get_device_info() {
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

  py::dict info;
  info["name"] = std::string(prop.name);
  info["compute_capability"] =
      std::to_string(prop.major) + "." + std::to_string(prop.minor);
  info["global_memory_mb"] = prop.totalGlobalMem / (1024 * 1024);
  info["shared_memory_per_block"] = prop.sharedMemPerBlock;
  info["max_threads_per_block"] = prop.maxThreadsPerBlock;
  info["warp_size"] = prop.warpSize;
  info["multiprocessor_count"] = prop.multiProcessorCount;
  info["clock_rate_mhz"] = prop.clockRate / 1000;
  info["memory_clock_rate_mhz"] = prop.memoryClockRate / 1000;
  info["memory_bus_width_bits"] = prop.memoryBusWidth;

  return info;
}

// ============================================================================
// Python Module Definition
// ============================================================================
PYBIND11_MODULE(mini_cublas_cpp, m) {
  m.doc() = "Mini cuBLAS — GPU-accelerated matrix operations via CUDA";

  // Matrix multiplication
  m.def("matmul_naive", &matmul_naive, "Naive SGEMM using global memory only",
        py::arg("A"), py::arg("B"));

  m.def("matmul_tiled", &matmul_tiled, "Tiled SGEMM using shared memory",
        py::arg("A"), py::arg("B"), py::arg("tile_size") = 16);

  m.def("cublas_matmul", &cublas_matmul,
        "cuBLAS SGEMM reference implementation", py::arg("A"), py::arg("B"));

  // Transpose
  m.def("transpose", &transpose_op,
        "Tiled transpose with bank conflict avoidance", py::arg("input"));

  // Dot product
  m.def("dot_product", &dot_product,
        "Dot product with tree reduction + warp shuffle", py::arg("a"),
        py::arg("b"));

  // Element-wise operations
  m.def("elementwise_add", &ew_add, "Element-wise addition: C = A + B",
        py::arg("A"), py::arg("B"));

  m.def("elementwise_scalar_mul", &ew_scalar_mul,
        "Element-wise scalar multiplication: C = A * scalar", py::arg("A"),
        py::arg("scalar"));

  // Benchmarking
  m.def("benchmark_matmul", &benchmark_matmul,
        "Benchmark all matmul implementations", py::arg("M"), py::arg("N"),
        py::arg("K"), py::arg("num_iters") = 20);

  // Device info
  m.def("get_device_info", &get_device_info, "Get GPU device properties");
}
