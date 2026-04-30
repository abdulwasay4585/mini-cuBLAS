import sys
import os
import numpy as np

# Add the build and python directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'build'))

try:
    from mini_cublas import MiniCuBLAS
except ImportError as e:
    print(f"Failed to import mini_cublas: {e}")
    sys.exit(1)

def run_32x32():
    print("="*60)
    print("Mini cuBLAS — 32x32 Matrix Operations Run")
    print("="*60)
    
    mcb = MiniCuBLAS()
    
    info = mcb.device_info()
    print("\n[Device Info]")
    for k, v in info.items():
        print(f"  {k}: {v}")
    
    N = 32
    print(f"\nInitializing {N}x{N} matrices...")
    np.random.seed(42)
    A = np.random.randn(N, N).astype(np.float32)
    B = np.random.randn(N, N).astype(np.float32)
    
    print("\n[1] Naive SGEMM")
    C_naive, meta = mcb.matmul_naive_gpu(A, B)
    for k, v in meta.items(): print(f"  {k}: {v}")
    print(f"  Result sum: {np.sum(C_naive):.4f}")

    print("\n[2] Tiled SGEMM (16x16)")
    C_tiled16, meta = mcb.matmul_tiled_gpu(A, B, tile_size=16)
    for k, v in meta.items(): print(f"  {k}: {v}")
    print(f"  Result sum: {np.sum(C_tiled16):.4f}")

    print("\n[3] Tiled SGEMM (32x32)")
    C_tiled32, meta = mcb.matmul_tiled_gpu(A, B, tile_size=32)
    for k, v in meta.items(): print(f"  {k}: {v}")
    print(f"  Result sum: {np.sum(C_tiled32):.4f}")
    
    print("\n[4] cuBLAS SGEMM")
    C_cublas, meta = mcb.matmul_cublas(A, B)
    for k, v in meta.items(): print(f"  {k}: {v}")
    print(f"  Result sum: {np.sum(C_cublas):.4f}")

    print("\n[5] Transpose (A^T)")
    A_T, meta = mcb.transpose_gpu(A)
    for k, v in meta.items(): print(f"  {k}: {v}")
    print(f"  Result sum: {np.sum(A_T):.4f} (matches A sum: {np.sum(A):.4f})")
    
    print("\n[6] Element-wise Add (A + B)")
    Add_res, meta = mcb.elementwise_add_gpu(A, B)
    for k, v in meta.items(): print(f"  {k}: {v}")
    print(f"  Result sum: {np.sum(Add_res):.4f}")
    
    print("\n[7] Element-wise Scalar Multiply (A * 2.5)")
    Mul_res, meta = mcb.elementwise_scalar_mul_gpu(A, 2.5)
    for k, v in meta.items(): print(f"  {k}: {v}")
    print(f"  Result sum: {np.sum(Mul_res):.4f}")
    
    print("\n[8] Dot Product")
    a_flat = A.flatten()
    b_flat = B.flatten()
    dot_res, meta = mcb.dot_product_gpu(a_flat, b_flat)
    for k, v in meta.items(): print(f"  {k}: {v}")
    print(f"  Result: {dot_res:.4f}")

    print("\n" + "="*60)
    print("All 32x32 operations completed successfully.")
    print("="*60)

if __name__ == "__main__":
    run_32x32()
