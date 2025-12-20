#include <cuda_runtime.h>
#include "gemm_kernels.h"
#include <cublas_v2.h>


__global__ void naive_gemm(float* A, float* B, float* C, int M, int K, int N){

	int i = threadIdx.y + blockIdx.y * blockDim.y;
	int j = threadIdx.x + blockIdx.x * blockDim.x;

	if (i < M && j < N){

		float tmp = 0.0;
		for (int k = 0; k<K; k++){
			tmp += A[i*K +k] * B[k*N+ j];
		}
		C[i*N + j] = tmp;
	}
	
}






void dgemm_cuda(float* d_A, float* d_B, float* d_C, int M, int K, int N){
    
	dim3 block(32, 32);
    dim3 grid((N + 32 - 1) / 32,
              (M + 32 - 1) / 32);

    naive_gemm<<<grid, block>>>(d_A, d_B, d_C, M, K, N);
}


cublasStatus_t cublas_sgemm(cublasHandle_t handle, float* A, float* B, float* C, float alpha, float beta, int M, int N, int K){

	cublasStatus_t status = 
					cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, 
											B, N,  // ldb
                                            A, K,  // lda
                                            &beta, 
                                            C, N); // ldc

	return status;
}
