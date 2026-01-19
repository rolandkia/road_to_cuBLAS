#ifndef GEMM_KERNELS_H
#define GEMM_KERNELS_H

#include <cublas_v2.h>
#include <string>

void sgemm_cuda(float* A, float* B, float* C, int M, int N, int K, std::string version);

cublasStatus_t cublas_sgemm(cublasHandle_t handle, float* A, float* B, float* C, float alpha, float beta, int M, int N, int K);

#endif	