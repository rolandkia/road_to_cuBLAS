#ifndef GEMM_KERNELS_H
#define GEMM_KERNELS_H

void dgemm_cuda(double* A, double* B, double* C, int M, int N, int K);

#endif