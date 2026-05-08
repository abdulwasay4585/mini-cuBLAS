"""
Page 4 - Operation Explorer
"""
import streamlit as st
import numpy as np
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
build_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'build')
if os.path.exists(build_dir):
    sys.path.insert(0, build_dir)
from mini_cublas import MiniCuBLAS

st.set_page_config(page_title="Operation Explorer", page_icon="", layout="wide")
st.markdown("# Operation Explorer")
st.markdown("**Interactive matrix operations with kernel metadata**")
st.markdown("---")

@st.cache_resource
def get_cublas():
    return MiniCuBLAS()
mcb = get_cublas()

operation = st.selectbox("Select Operation", [
    "Matrix Multiplication (SGEMM)", "Transpose",
    "Element-wise Add", "Dot Product", "Scalar Multiply"
])
st.markdown("---")

col_in, col_out = st.columns(2)
with col_in:
    st.markdown("### Input")
    rows_a = st.number_input("Rows A", 2, 16, 4)
    cols_a = st.number_input("Cols A", 2, 16, 4)
    np.random.seed(42)
    A = np.random.randint(0, 10, (rows_a, cols_a)).astype(np.float32)

    if operation == "Matrix Multiplication (SGEMM)":
        cols_b = st.number_input("Cols B", 2, 16, 4)
        B = np.random.randint(0, 10, (cols_a, cols_b)).astype(np.float32)
    elif operation in ["Element-wise Add"]:
        B = np.random.randint(0, 10, (rows_a, cols_a)).astype(np.float32)
    elif operation == "Dot Product":
        B = np.random.randint(0, 10, (rows_a, cols_a)).astype(np.float32)
    elif operation == "Scalar Multiply":
        scalar_val = st.number_input("Scalar", value=2.0, step=0.5)

    st.markdown("**Matrix A:**")
    st.dataframe(A, use_container_width=True)
    if operation not in ["Transpose", "Scalar Multiply"]:
        st.markdown("**Matrix B:**")
        st.dataframe(B, use_container_width=True)

with col_out:
    st.markdown("### Result")
    if st.button("Compute", type="primary", use_container_width=True):
        try:
            if operation == "Matrix Multiplication (SGEMM)":
                result, meta = mcb.matmul_tiled_gpu(A, B, 16)
                st.markdown("**C = A x B:**")
                st.dataframe(np.round(result, 2), use_container_width=True)
                expected = A @ B
                err = np.max(np.abs(expected - result))
                if err < 1e-3:
                    st.success(f"Verified (err: {err:.2e})")
                else:
                    st.error(f"err: {err:.2e}")
                M, K = A.shape; N = B.shape[1]
                st.markdown("### Kernel Metadata")
                for k, v in meta.items(): st.text(f"{k}: {v}")
                st.text(f"Naive reads: {2*M*N*K*4:,} bytes | Tiled reads: {2*M*N*K*4//16:,} bytes")

            elif operation == "Transpose":
                result, meta = mcb.transpose_gpu(A)
                st.markdown("**Aᵀ:**")
                st.dataframe(np.round(result, 2), use_container_width=True)
                err = np.max(np.abs(A.T - result))
                if err < 1e-5:
                    st.success(f"Verified (err: {err:.2e})")
                else:
                    st.error(f"err: {err:.2e}")
                for k, v in meta.items(): st.text(f"{k}: {v}")

            elif operation == "Element-wise Add":
                result, meta = mcb.elementwise_add_gpu(A, B)
                st.markdown("**C = A + B:**")
                st.dataframe(np.round(result, 2), use_container_width=True)
                err = np.max(np.abs((A + B) - result))
                if err < 1e-5:
                    st.success("Verified")
                else:
                    st.error(f"err: {err:.2e}")
                for k, v in meta.items(): st.text(f"{k}: {v}")

            elif operation == "Dot Product":
                a_f, b_f = A.flatten(), B.flatten()
                result, meta = mcb.dot_product_gpu(a_f, b_f)
                expected = float(np.dot(a_f, b_f))
                st.markdown(f"**Dot Product = `{result:.4f}`**")
                err = abs(expected - result)
                if err < 1e-2:
                    st.success(f"NumPy={expected:.4f}, err={err:.2e}")
                else:
                    st.error(f"err: {err:.2e}")
                for k, v in meta.items(): st.text(f"{k}: {v}")
                st.info("Uses tree reduction + `__shfl_down_sync()` warp shuffle")

            elif operation == "Scalar Multiply":
                result, meta = mcb.elementwise_scalar_mul_gpu(A, scalar_val)
                st.markdown(f"**C = A x {scalar_val}:**")
                st.dataframe(np.round(result, 2), use_container_width=True)
                err = np.max(np.abs(A * scalar_val - result))
                if err < 1e-5:
                    st.success("Verified")
                else:
                    st.error(f"err: {err:.2e}")
                for k, v in meta.items(): st.text(f"{k}: {v}")
        except Exception as e:
            st.error(f"Error: {e}")

st.markdown("---")
st.caption("Mini cuBLAS - Operation Explorer - Abdul Wasay")
