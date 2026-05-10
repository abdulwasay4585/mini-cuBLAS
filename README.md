# Mini cuBLAS Project

## What is this project?
Welcome to the Mini cuBLAS project! This is an educational and high performance library for GPU acceleration. 

In simple terms, computers usually process math using the CPU (Central Processing Unit). However, for massive math problems like Matrix Multiplication or Deep Learning, the CPU is too slow. The GPU (Graphics Processing Unit) is much faster because it has thousands of tiny cores that can do math at the exact same time.

This project writes raw, low level code in C++ and NVIDIA CUDA to make the GPU do linear algebra math as fast as possible. Then, it connects that super fast C++ code to Python, so anyone can easily use it just like they use normal Python libraries.

## Detailed Core Features

### 1. Matrix Multiplication (GEMM)
Multiplying two matrices is the foundation of Artificial Intelligence.
* Naive Approach: We built a simple version that reads directly from the main GPU memory. It works, but it is slow because main memory is far away from the processing cores.
* Tiled Shared Memory Approach: We built a highly advanced version that loads small chunks (tiles) of the matrices into a super fast cache called Shared Memory. This makes the math incredibly fast.

### 2. Matrix Transpose
Flipping a matrix over its diagonal is tricky for a GPU because it causes memory traffic jams called Bank Conflicts. We fixed this by adding a tiny bit of invisible padding to the shared memory, making the data flow perfectly.

### 3. Dot Product (Vector Math)
When multiplying two vectors, we use a technique called Tree Reduction and Warp Shuffles. Instead of one core adding all the numbers, pairs of cores add numbers together in a tree shape, finishing the job in a fraction of the time.

### 4. Elementwise Operations
We included basic operations that add numbers or apply Neural Network activation functions like ReLU and Sigmoid to every single number in a matrix instantly.

### 5. Python Integration (pybind11)
Nobody wants to write complex C++ code every day. We used a tool called pybind11 to wrap our fast C++ code into a simple Python package.

### 6. Interactive Web Application
We built a beautiful web interface using Streamlit. It includes:
* Memory Hierarchy Tutorial: Explains how GPU memory works.
* Tiling Simulator: A step by step visualizer showing exactly how the GPU processes matrix tiles.
* Benchmark Page: A chart that compares our custom code speed against NVIDIA official code speed.

## How the Folders are Organized

* csrc directory: This folder holds all the low level C++ and CUDA source code. This is where the actual math happens.
* python directory: This folder holds the Streamlit web application and the python scripts that make the interface work.
* test directory: This folder holds automated tests that check our GPU math against Python math to make sure the answers are exactly correct.

## Step by Step Build Instructions

To use this project, you need to compile the C++ code into a Python library. 

1. First, make sure you have CMake and the NVIDIA CUDA Toolkit installed.
2. Open your terminal and go into the project folder.
3. Create a folder named build using the mkdir command.
4. Go into the build folder using the cd command.
5. Run the cmake command to configure the project.
6. Run the make command to compile the code.
7. Tell Python where to find the new library by exporting the build directory to your PYTHONPATH.

## How to Run the Web Application

Once the code is built, you can start the visualizer!

1. Make sure you install the required Python tools like streamlit, plotly, and numpy using pip.
2. Start the application by running the Streamlit app.py script.
3. A web browser will open automatically showing the dashboard.

## Conclusion
This project demonstrates how to bridge the gap between extremely fast, complex hardware programming and easy, accessible Python web development.

## Quick Start Commands

```bash
# Clone and build
git clone https://github.com/abdulwasay4585/mini-cuBLAS.git
cd mini-cuBLAS
mkdir build && cd build
cmake ..
make -j$(nproc)
export PYTHONPATH=$(pwd):$PYTHONPATH

# Run tests
cd ..
pytest test/test_correctness.py -v

# Launch visualizer
pip install streamlit plotly matplotlib numpy
streamlit run python/app.py
```
