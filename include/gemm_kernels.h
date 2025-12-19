#ifndef GEMM_KERNELS_H
#define GEMM_KERNELS_H

void dgemm_cuda(float* A, float* B, float* C, int M, int N, int K);

#endif	