# ============================================================================
# Performance Notes (May 9):
#   - Tested with SM 8.6 (RTX A2000 Ampere)
#   - --use_fast_math provides ~15% speedup on tiled SGEMM
#   - -O3 enables aggressive inlining of small device functions
# ============================================================================

cmake_minimum_required(VERSION 3.18)
project(mini_cublas LANGUAGES CXX CUDA)

# ============================================================================
# CUDA Configuration
# ============================================================================
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

# Target SM 8.6 (RTX A2000 — Ampere)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

# ============================================================================
# Find Dependencies
# ============================================================================
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)

# Find cuBLAS
find_library(CUBLAS_LIB cublas
    HINTS /usr/lib/x86_64-linux-gnu /usr/local/cuda/lib64)

# ============================================================================
# Collect CUDA source files
# ============================================================================
set(CUDA_SOURCES
    csrc/kernels/naive_gemm.cu
    csrc/kernels/tiled_gemm.cu
    csrc/kernels/transpose.cu
    csrc/kernels/dot_product.cu
    csrc/kernels/elementwise.cu
)

# ============================================================================
# Build the pybind11 Python extension module
# ============================================================================
pybind11_add_module(mini_cublas_cpp
    csrc/bindings.cpp
    ${CUDA_SOURCES}
)

# Include directories
target_include_directories(mini_cublas_cpp PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc
    ${Python3_INCLUDE_DIRS}
)

# Link cuBLAS
if(CUBLAS_LIB)
    target_link_libraries(mini_cublas_cpp PRIVATE ${CUBLAS_LIB})
    message(STATUS "Found cuBLAS: ${CUBLAS_LIB}")
else()
    message(WARNING "cuBLAS not found — cublas_matmul will not be available")
endif()

# CUDA flags for better performance
target_compile_options(mini_cublas_cpp PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        -O3
        --use_fast_math
    >
)

# ============================================================================
# Output info
# ============================================================================
message(STATUS "Python: ${Python3_EXECUTABLE}")
message(STATUS "pybind11: ${pybind11_INCLUDE_DIR}")
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
