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

#define BLOCKSIZE 32

__global__ void sgemm_coalesed(float* A, float* B, float* C, int M, int K, int N){

	int i = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
    int j = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

    if (i < M && j < N){
        float tmp = 0.0;
        for (int k = 0; k < K; k++){
            // A[i][k] * B[k][j]
            tmp += A[i * K + k] * B[k * N + j];
        }
        C[i * N + j] = tmp;
    }
}

__global__ void shared_memory_matrix_mul(float* A, float* B, float* C, int M, int K, int N) {

	__shared__ float As[BLOCKSIZE][BLOCKSIZE];
    __shared__ float Bs[BLOCKSIZE][BLOCKSIZE];

    int threadRow = threadIdx.y;
    int threadCol = threadIdx.x;
    
    int row = blockIdx.y * BLOCKSIZE + threadRow;
    int col = blockIdx.x * BLOCKSIZE + threadCol;

    float tmp = 0.0;

    for (int bkIdx = 0; bkIdx < (K + BLOCKSIZE - 1) / BLOCKSIZE; ++bkIdx) {
        
        if (row < M && (bkIdx * BLOCKSIZE + threadCol) < K) {
            As[threadRow][threadCol] = A[row * K + bkIdx * BLOCKSIZE + threadCol];
        } else {
            As[threadRow][threadCol] = 0.0;
        }

        if (col < N && (bkIdx * BLOCKSIZE + threadRow) < K) {
            Bs[threadRow][threadCol] = B[(bkIdx * BLOCKSIZE + threadRow) * N + col];
        } else {
            Bs[threadRow][threadCol] = 0.0;
        }

        __syncthreads();

        for (int dotIdx = 0; dotIdx < BLOCKSIZE; ++dotIdx) {
            tmp += As[threadRow][dotIdx] * Bs[dotIdx][threadCol];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = tmp;
    }
}


#define BM 64   // Tile height in C
#define BN 64   // Tile width in C
#define BK 8    // Tile depth (K)
#define TM 8    // Rows per thread (register tiling)

__global__ void matrixMul1DRegisterTiling(const float* __restrict__ A, const float* __restrict__ B,
    float* __restrict__ C, int M, int K, int N) 
{
    // Shared memory
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    // Thread mapping
    // 512 threads per block
    int tid = threadIdx.x;
    int threadCol = tid % BN;        // column in C
    int threadRowBlock = tid / BN;   // vertical block
    int rowBase = threadRowBlock * TM;

    // Block base pointers
    int blockRow = blockIdx.y * BM;
    int blockCol = blockIdx.x * BN;

    // Register accumulator
    float acc[TM] = {0.0f};

    // Loop over K tiles
    for (int kb = 0; kb < K; kb += BK) {

        int aIdx = tid;
        if (aIdx < BM * BK) {
            int aRow = aIdx / BK;
            int aCol = aIdx % BK;

            int globalRow = blockRow + aRow;
            int globalCol = kb + aCol;

            As[aRow * BK + aCol] =
                (globalRow < M && globalCol < K)
                ? A[globalRow * K + globalCol]
                : 0.0f;
        }

        int bIdx = tid;
        if (bIdx < BK * BN) {
            int bRow = bIdx / BN;
            int bCol = bIdx % BN;

            int globalRow = kb + bRow;
            int globalCol = blockCol + bCol;

            Bs[bRow * BN + bCol] =
                (globalRow < K && globalCol < N)
                ? B[globalRow * N + globalCol]
                : 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            float b = Bs[k * BN + threadCol];
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                acc[i] += As[(rowBase + i) * BK + k] * b;
            }
        }

        __syncthreads();
    }

    #pragma unroll 
    for (int i = 0; i < TM; ++i) {
        int globalRow = blockRow + rowBase + i;
        int globalCol = blockCol + threadCol;

        if (globalRow < M && globalCol < N) {
            C[globalRow * N + globalCol] = acc[i];
        }
    }
}




void dgemm_cuda(float* d_A, float* d_B, float* d_C, int M, int K, int N){
    
	// dim3 gridDim((M + 32 -1)/32, 
	// 			(N + 32 -1)/32);
	// dim3 blockDim(32*32);

	// sgemm_coalesed<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);


	// dim3 dimBlock(BLOCKSIZE, BLOCKSIZE); 
	// dim3 dimGrid((N + BLOCKSIZE - 1) / BLOCKSIZE, (M + BLOCKSIZE - 1) / BLOCKSIZE);
	// shared_memory_matrix_mul<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, M, K, N);
	

	// dim3 dimBlock(BN * (BM / TM));
	// dim3 dimGrid((N + BN - 1) / BN, (M + BM - 1) / BM);
	// matrixMul1DRegisterTiling<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, M, K, N);


	dim3 blockDim((BM / TM) * BN);   // 512 threads
	dim3 gridDim(
    (N + BN - 1) / BN,
    (M + BM - 1) / BM);

	matrixMul1DRegisterTiling<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);

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
