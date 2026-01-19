# CUDA SGEMM Optimization Suite

This project implements and benchmarks various optimization techniques for SGEMM (Single-precision General Matrix Multiplication) on NVIDIA GPUs using CUDA. Starting from a naive implementation, the project evolves through several optimization stages to approach the performance of industry-standard libraries like cuBLAS.

## Project Goal

The primary objective is to demonstrate how memory hierarchy management, data tiling, and instruction-level parallelism can drastically improve throughput in compute-bound kernels.

## Optimization Journey

Each kernel in this suite represents a specific optimization technique:
- **Memory Coalescing**: Reorganizing global memory access patterns to ensure contiguous threads access contiguous memory addresses.
- **Shared Memory Tiling (1D & 2D)**: Loading data into on-chip Shared Memory to reduce redundant high-latency DRAM fetches.
- **Thread Tiling**: Increasing the work-per-thread to improve register reuse and reduce the instruction overhead.
- **Vectorization (float4)**: Utilizing 128-bit memory transactions to saturate the GPU's memory bandwidth.
- **Double Buffering**: Overlapping data movement with computation by prefetching the next data tile while calculating the current one.


### Compilation

```bash
make              # Compile kernel
make clean        # Remove binaries
```

### Usage

Run the tool by specifying the kernel method and the matrix size (N×N):
```bash
./gemm_tool -m <method> -s <size> [-t]
```
-

### Options:
- `-m <method>` : Specifies the GPU kernel to benchmark. Choose from the available methods listed above (e.g., `sgemm_vectorized_double_buffering`).
- `-s <size>` : Sets the dimension N for the square matrices (N×N). Defaults to **2048**. Note: GPU results are automatically verified against cuBLAS.
- `-t` : Enables **Unit Testing**. Compares the GPU output against a CPU reference implementation to ensure mathematical correctness.

### Example Command
To run the most optimized version on 4096x4096 matrices:
```bash
./gemm_tool -m sgemm_vectorized_double_buffering -s 4096
```


## Implemented Kernels

- `sgemm_naive`: Basic implementation without optimizations.

- `sgemm_coalescing`: Improved global memory access.

- `sgemm_tiled`: Basic 1D shared memory tiling.

- `sgemm_2d_tiled`: 2D block tiling for better data reuse.

- `sgemm_vectorized_2d_tiled`: Uses float4 for high-speed memory transactions.

- `sgemm_vectorized_2d_tiled_safe`: Uses float4 for high-speed memory transactions safe version.

- `sgemm_vectorized_double_buffering`: The most advanced version, hiding memory latency through prefetching.
