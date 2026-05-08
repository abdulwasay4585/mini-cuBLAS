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
    st.markdown("**Matrix A:**")
    st.dataframe(A, use_container_width=True)

with col_out:
    st.markdown("### Result")
    st.info("Compute button coming soon")

st.markdown("---")
st.caption("Mini cuBLAS - Operation Explorer - Abdul Wasay")
