#include <cuda_runtime.h>

__global__ void dgemm_kernel(const double* A, const double* B, double* C, int M, int K, int N){

	int i = threadIdx.y + blockIdx.y * blockDim.y
	int j = threadIdx.x + blockIdx.x * blockDim.x

	if (i < M && j < N){

		double tmp = 0.0;
		for (int k = 0; k<K; k++){
			tmp += A[i*K +k] * B[k*N+ j]
		}
		C[i*K + j] = tmp;
	}
	
}


void dgemm_cuda(const double* d_A, const double* d_B, double* d_C, int M, int K, int N){
    
	dim3 block(32, 32);
    dim3 grid((N + 32 - 1) / 32,
              (M + 32 - 1) / 32);

    dgemm_kernel<<<grid, block>>>(d_A, d_B, d_C, M, K, N);
}

