#include <cuda_runtime.h>
#include "gemm_kernels.h"
#include <cublas_v2.h>
#include <assert.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

__global__ void naive_gemm(float* A, float* B, float* C, int M, int K, int N)
{

	int j = threadIdx.y + blockIdx.y * blockDim.y;
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	if (i < M && j < N){

		float tmp = 0.0;
		for (int k = 0; k<K; k++){
			tmp += A[i*K +k] * B[k*N+ j];
		}
		C[i*N + j] = tmp;
	}
	
}

__global__ void coalescing_sgemm(float* A, float* B, float* C, int M, int K, int N)
{

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

template<const int TILE_SIZE>
__global__ void tiled_sgemm(float* A, float* B, float* C, int M, int K, int N)
{

	__shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

	int row = blockIdx.y * TILE_SIZE + threadIdx.y;
	int col = blockIdx.x * TILE_SIZE + threadIdx.x;

	float tmp = 0.0;
	for (int t = 0; t < CEIL_DIV(K, TILE_SIZE); ++t) {
        
        if (row < M && (t * TILE_SIZE + threadIdx.x) < K)
            tileA[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
        else
            tileA[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < N && (t * TILE_SIZE + threadIdx.y) < K)
            tileB[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        else
            tileB[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k) {
            tmp += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }

		__syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = tmp;
    }

}

template <const int BM, const int BN, const int BK, const int TM>
__global__ void sgemm_1D_blocktiling(float*  A,  float*  B, float*  C, int M, int K, int N) 
{

	__shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

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

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void sgemm_2D_blocktiling(float * A, float * B, float * C, int M, int K, int N)
{
    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;

    constexpr uint THREADS_PER_BLOCK = (BM * BN) / (TM * TN);

    const uint threadCol = threadIdx.x % (BN / TN);
    const uint threadRow = threadIdx.x / (BN / TN);

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    const uint blockRow = cRow * BM;
    const uint blockCol = cCol * BN;

    const uint innerRowA = threadIdx.x / BK;
    const uint innerColA = threadIdx.x % BK;
    const uint strideA   = THREADS_PER_BLOCK / BK;

    const uint innerRowB = threadIdx.x / BN;
    const uint innerColB = threadIdx.x % BN;
    const uint strideB   = THREADS_PER_BLOCK / BN;

    float threadResults[TM * TN] = {0.0f};
    float regM[TM];
    float regN[TN];

    for (uint bk = 0; bk < K; bk += BK) {

        for (uint loadOffset = 0; loadOffset < BM; loadOffset += strideA) {
            uint row = blockRow + innerRowA + loadOffset;
            uint col = bk + innerColA;

            if (row < M && col < K)
                As[(innerRowA + loadOffset) * BK + innerColA] =
                    A[row * K + col];
            else
                As[(innerRowA + loadOffset) * BK + innerColA] = 0.0f;
        }

        for (uint loadOffset = 0; loadOffset < BK; loadOffset += strideB) {
            uint row = bk + innerRowB + loadOffset;
            uint col = blockCol + innerColB;

            if (row < K && col < N)
                Bs[(innerRowB + loadOffset) * BN + innerColB] =
                    B[row * N + col];
            else
                Bs[(innerRowB + loadOffset) * BN + innerColB] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (uint dot = 0; dot < BK; ++dot) {

            #pragma unroll
            for (uint i = 0; i < TM; ++i)
                regM[i] = As[(threadRow * TM + i) * BK + dot];

            #pragma unroll
            for (uint i = 0; i < TN; ++i)
                regN[i] = Bs[dot * BN + threadCol * TN + i];

            #pragma unroll
            for (uint i = 0; i < TM; ++i)
                #pragma unroll
                for (uint j = 0; j < TN; ++j)
                    threadResults[i * TN + j] += regM[i] * regN[j];
        }

        __syncthreads();
    }

    #pragma unroll
    for (uint i = 0; i < TM; ++i) {
        uint row = blockRow + threadRow * TM + i;
        if (row < M) {
            #pragma unroll
            for (uint j = 0; j < TN; ++j) {
                uint col = blockCol + threadCol * TN + j;
                if (col < N)
                    C[row * N + col] = threadResults[i * TN + j];
            }
        }
    }
}

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void sgemm_vectorized_2D_blocktiling(float * A, float * B, float * C, int M, int K, int N)
{
    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;

    constexpr uint THREADS_PER_BLOCK = (BM * BN) / (TM * TN);

    const uint threadCol = threadIdx.x % (BN / TN);
    const uint threadRow = threadIdx.x / (BN / TN);

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    const uint blockRow = cRow * BM;
    const uint blockCol = cCol * BN;

    const uint innerRowA = threadIdx.x / (BK/4);
    const uint innerColA = threadIdx.x % (BK/4);
    const uint strideA   = THREADS_PER_BLOCK / (BK/4);

    const uint innerRowB = threadIdx.x / (BN/4);
    const uint innerColB = threadIdx.x % (BN/4);
    const uint strideB   = THREADS_PER_BLOCK / (BN/4);

    float threadResults[TM * TN] = {0.0f};
    float regM[TM];
    float regN[TN];

    for (uint bk = 0; bk < K; bk += BK) {

        for (uint loadOffset = 0; loadOffset < BM; loadOffset += strideA) {
            uint row = blockRow + innerRowA + loadOffset;
            uint col = bk + (innerColA*4);


			float4 tmp = (row < M && col < K) ? 
            	reinterpret_cast<float4*>(&A[row * K + col])[0] : 
            	make_float4(0.0f, 0.0f, 0.0f, 0.0f);

			uint sharedIdx = (innerRowA + loadOffset) * BK + (innerColA * 4);
			As[sharedIdx + 0] = tmp.x;
			As[sharedIdx + 1] = tmp.y;
			As[sharedIdx + 2] = tmp.z;
			As[sharedIdx + 3] = tmp.w;

        }

        for (uint loadOffset = 0; loadOffset < BK; loadOffset += strideB) {
            uint row = bk + innerRowB + loadOffset;
            uint col = blockCol + (innerColB*4);

			float4 tmp = (row < K && col < N) ? 
            	reinterpret_cast<float4*>(&B[row * N + col])[0] : 
           		make_float4(0.0f, 0.0f, 0.0f, 0.0f);


			uint sharedIdx = (innerRowB + loadOffset) * BN + (innerColB * 4);
			Bs[sharedIdx + 0] = tmp.x;
			Bs[sharedIdx + 1] = tmp.y;
			Bs[sharedIdx + 2] = tmp.z;
			Bs[sharedIdx + 3] = tmp.w;
        }

        __syncthreads();

        #pragma unroll
        for (uint dot = 0; dot < BK; ++dot) {

            #pragma unroll
            for (uint i = 0; i < TM; ++i)
                regM[i] = As[(threadRow * TM + i) * BK + dot];

            #pragma unroll
            for (uint i = 0; i < TN; i+=4)
                // regN[i] = Bs[dot * BN + threadCol * TN + i];
				reinterpret_cast<float4*>(&regN[i])[0] = 
            		reinterpret_cast<float4*>(&Bs[dot * BN + threadCol * TN + i])[0];

            #pragma unroll
            for (uint i = 0; i < TM; ++i)
                #pragma unroll
                for (uint j = 0; j < TN; ++j)
                    threadResults[i * TN + j] += regM[i] * regN[j];
        }

        __syncthreads();
    }

    #pragma unroll
    for (uint i = 0; i < TM; ++i) {
        uint row = blockRow + threadRow * TM + i;
        if (row < M) {
            #pragma unroll
            for (uint j = 0; j < TN; ++j) {
                uint col = blockCol + threadCol * TN + j;
                if (col < N)
                    C[row * N + col] = threadResults[i * TN + j];
            }
        }
    }
}

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void sgemm_vectorized_2D_blocktiling_safe(float * A, float * B, float * C, int M, int K, int N)
{
    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;

    constexpr uint THREADS_PER_BLOCK = (BM * BN) / (TM * TN);

    const uint threadCol = threadIdx.x % (BN / TN);
    const uint threadRow = threadIdx.x / (BN / TN);

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    const uint blockRow = cRow * BM;
    const uint blockCol = cCol * BN;

    const uint innerRowA = threadIdx.x / (BK/4);
    const uint innerColA = threadIdx.x % (BK/4);
    const uint strideA   = THREADS_PER_BLOCK / (BK/4);

    const uint innerRowB = threadIdx.x / (BN/4);
    const uint innerColB = threadIdx.x % (BN/4);
    const uint strideB   = THREADS_PER_BLOCK / (BN/4);

    float threadResults[TM * TN] = {0.0f};
    float regM[TM];
    float regN[TN];

    for (uint bk = 0; bk < K; bk += BK) {

        for (uint loadOffset = 0; loadOffset < BM; loadOffset += strideA) {
            uint row = blockRow + innerRowA + loadOffset;
            uint col = bk + (innerColA*4);

			if (row < M && col + 3 < K && (reinterpret_cast<uintptr_t>(&A[row * K + col]) % 16 == 0)) {
				reinterpret_cast<float4*>(&As[(innerRowA + loadOffset) * BK + innerColA * 4])[0] = 
					reinterpret_cast<float4*>(&A[row * K + col])[0];
			} 
			else {
				for (int i = 0; i < 4; ++i) {
					if (row < M && (col + i) < K) {
						As[(innerRowA + loadOffset) * BK + innerColA * 4 + i] = A[row * K + col + i];
					} 
					else {
						As[(innerRowA + loadOffset) * BK + innerColA * 4 + i] = 0.0f;
					}
				}
    		}

        }

        for (uint loadOffset = 0; loadOffset < BK; loadOffset += strideB) {
            uint row = bk + innerRowB + loadOffset;
            uint col = blockCol + (innerColB*4);

			if (row < K && col + 3 < N && (reinterpret_cast<uintptr_t>(&B[row * N + col]) % 16 == 0)) {
				reinterpret_cast<float4*>(&Bs[(innerRowB + loadOffset) * BN + innerColB * 4])[0] = 
					reinterpret_cast<float4*>(&B[row * N + col])[0];
			} 
			else {
				for (int i = 0; i < 4; ++i) {
					if (row < K && (col + i) < N) {
						Bs[(innerRowB + loadOffset) * BN + innerColB * 4 + i] = B[row * N + col + i];
					} 
					else {
						Bs[(innerRowB + loadOffset) * BN + innerColB * 4 + i] = 0.0f;
					}
				}
			}

        }

        __syncthreads();

        #pragma unroll
        for (uint dot = 0; dot < BK; ++dot) {

            #pragma unroll
            for (uint i = 0; i < TM; ++i)
                regM[i] = As[(threadRow * TM + i) * BK + dot];

            #pragma unroll
            for (uint i = 0; i < TN; i += 4)
                // regN[i] = Bs[dot * BN + threadCol * TN + i];
				reinterpret_cast<float4*>(&regN[i])[0] = 
            		reinterpret_cast<float4*>(&Bs[dot * BN + threadCol * TN + i])[0];


            #pragma unroll
            for (uint i = 0; i < TM; ++i)
                #pragma unroll
                for (uint j = 0; j < TN; ++j)
                    threadResults[i * TN + j] += regM[i] * regN[j];
        }

        __syncthreads();
    }

    #pragma unroll
    for (uint i = 0; i < TM; ++i) {
        uint row = blockRow + threadRow * TM + i;
        if (row < M) {
            #pragma unroll
            for (uint j = 0; j < TN; ++j) {
                uint col = blockCol + threadCol * TN + j;
                if (col < N)
                    C[row * N + col] = threadResults[i * TN + j];
            }
        }
    }
}

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void sgemm_vectorized_double_buffering(float * A, float * B, float * C, int M, int K, int N) {
    
	const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;

    constexpr uint THREADS_PER_BLOCK = (BM * BN) / (TM * TN);

    const uint threadCol = threadIdx.x % (BN / TN);
    const uint threadRow = threadIdx.x / (BN / TN);

    __shared__ float As[2][BM * BK];
    __shared__ float Bs[2][BK * BN];

    const uint blockRow = cRow * BM;
    const uint blockCol = cCol * BN;

    const uint innerRowA = threadIdx.x / (BK / 4);
    const uint innerColA = threadIdx.x % (BK / 4);
    const uint strideA   = THREADS_PER_BLOCK / (BK / 4);

    const uint innerRowB = threadIdx.x / (BN / 4);
    const uint innerColB = threadIdx.x % (BN / 4);
    const uint strideB   = THREADS_PER_BLOCK / (BN / 4);

    float threadResults[TM * TN] = {0.0f};
    float regM[TM];
    float regN[TN];

    uint writeIdx = 0; 
    {
        uint bk = 0;
        for (uint loadOffset = 0; loadOffset < BM; loadOffset += strideA) {
            uint row = blockRow + innerRowA + loadOffset;
            uint col = bk + (innerColA * 4);
            float4 tmp = (row < M && col < K) ? reinterpret_cast<float4*>(&A[row * K + col])[0] : make_float4(0,0,0,0);
            uint sIdx = (innerRowA + loadOffset) * BK + (innerColA * 4);
            As[writeIdx][sIdx + 0] = tmp.x; 
			As[writeIdx][sIdx + 1] = tmp.y;
            As[writeIdx][sIdx + 2] = tmp.z; 
			As[writeIdx][sIdx + 3] = tmp.w;
        }
        for (uint loadOffset = 0; loadOffset < BK; loadOffset += strideB) {
            uint row = bk + innerRowB + loadOffset;
            uint col = blockCol + (innerColB * 4);
            float4 tmp = (row < K && col < N) ? reinterpret_cast<float4*>(&B[row * N + col])[0] : make_float4(0,0,0,0);
            uint sIdx = (innerRowB + loadOffset) * BN + (innerColB * 4);
            Bs[writeIdx][sIdx + 0] = tmp.x; 
			Bs[writeIdx][sIdx + 1] = tmp.y;
            Bs[writeIdx][sIdx + 2] = tmp.z; 
			Bs[writeIdx][sIdx + 3] = tmp.w;
        }
    }
    __syncthreads();

    uint readIdx = 0;
    for (uint bk = BK; bk < K; bk += BK) {

		writeIdx = (readIdx + 1) % 2;

        // CHARGEMENT ASYNCHRONE (Tuile suivante)
        for (uint loadOffset = 0; loadOffset < BM; loadOffset += strideA) {
            uint row = blockRow + innerRowA + loadOffset;
            uint col = bk + (innerColA * 4);
            float4 tmp = (row < M && col < K) ? reinterpret_cast<float4*>(&A[row * K + col])[0] : make_float4(0,0,0,0);
            uint sIdx = (innerRowA + loadOffset) * BK + (innerColA * 4);
            As[writeIdx][sIdx + 0] = tmp.x; 
			As[writeIdx][sIdx + 1] = tmp.y;
            As[writeIdx][sIdx + 2] = tmp.z; 
			As[writeIdx][sIdx + 3] = tmp.w;
        }
        for (uint loadOffset = 0; loadOffset < BK; loadOffset += strideB) {
            uint row = bk + innerRowB + loadOffset;
            uint col = blockCol + (innerColB * 4);
            float4 tmp = (row < K && col < N) ? reinterpret_cast<float4*>(&B[row * N + col])[0] : make_float4(0,0,0,0);
            uint sIdx = (innerRowB + loadOffset) * BN + (innerColB * 4);
            Bs[writeIdx][sIdx + 0] = tmp.x; 
			Bs[writeIdx][sIdx + 1] = tmp.y;
            Bs[writeIdx][sIdx + 2] = tmp.z; 
			Bs[writeIdx][sIdx + 3] = tmp.w;
        }

        // CALCUL (Tuile actuelle)
        #pragma unroll
        for (uint dot = 0; dot < BK; ++dot) {
            #pragma unroll
            for (uint i = 0; i < TM; ++i)
                regM[i] = As[readIdx][(threadRow * TM + i) * BK + dot];
            #pragma unroll
            for (uint i = 0; i < TN; i += 4)
                reinterpret_cast<float4*>(&regN[i])[0] = reinterpret_cast<float4*>(&Bs[readIdx][dot * BN + threadCol * TN + i])[0];
            #pragma unroll
            for (uint i = 0; i < TM; ++i)
                for (uint j = 0; j < TN; ++j)
                    threadResults[i * TN + j] += regM[i] * regN[j];
        }

        __syncthreads();
        readIdx = writeIdx;
    }

    #pragma unroll
    for (uint dot = 0; dot < BK; ++dot) {
        #pragma unroll
        for (uint i = 0; i < TM; ++i)
            regM[i] = As[readIdx][(threadRow * TM + i) * BK + dot];
        #pragma unroll
        for (uint i = 0; i < TN; i += 4)
            reinterpret_cast<float4*>(&regN[i])[0] = reinterpret_cast<float4*>(&Bs[readIdx][dot * BN + threadCol * TN + i])[0];
        #pragma unroll
        for (uint i = 0; i < TM; ++i)
            for (uint j = 0; j < TN; ++j)
                threadResults[i * TN + j] += regM[i] * regN[j];
    }

    #pragma unroll
    for (uint i = 0; i < TM; ++i) {
        uint row = blockRow + threadRow * TM + i;
        if (row < M) {
            #pragma unroll
            for (uint j = 0; j < TN; ++j) {
                uint col = blockCol + threadCol * TN + j;
                if (col < N) C[row * N + col] = threadResults[i * TN + j];
            }
        }
    }
}


void sgemm_cuda(float* d_A, float* d_B, float* d_C, int M, int K, int N, std::string version){


	if (version == "sgemm_naive"){
		dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
		dim3 blockDim(32, 32, 1);
		naive_gemm<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
		cudaDeviceSynchronize();
	}
	else if (version == "sgemm_coalescing"){
		dim3 gridDim(CEIL_DIV(N, 32), CEIL_DIV(M, 32));
		dim3 blockDim(32, 32, 1);
		coalescing_sgemm<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
		cudaDeviceSynchronize();
	}
	else if (version == "sgemm_tiled"){

		const int TILE_SIZE = 32;

		dim3 blockDim(TILE_SIZE, TILE_SIZE);
		dim3 gridDim(CEIL_DIV(N, TILE_SIZE), CEIL_DIV(M, TILE_SIZE));
		
		tiled_sgemm<TILE_SIZE><<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
		cudaDeviceSynchronize();

	}
	else if (version == "sgemm_1d_tiled"){

		const int BM = 64; 
		const int BN = 64; 
		const int BK = 8;
		const int TM = 8;

		dim3 blockDim((BM / TM) * BN); 
		dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));

		sgemm_1D_blocktiling<BM, BN, BK, TM><<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
		cudaDeviceSynchronize();

	}
	else if (version == "sgemm_2d_tiled"){

		const int BM = 128;
		const int BN = 128;
		const int BK = 8;
		const int TM = 8;
		const int TN = 8;

		dim3 blockDim((BN*BM)/(TN*TM));
		dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
		sgemm_2D_blocktiling<BM, BN, BK, TM, TN><<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
		cudaDeviceSynchronize();

	}
	else if (version == "sgemm_vectorized_2d_tiled"){
		
		const int BM = 128;
		const int BN = 128;
		const int BK = 8;
		const int TM = 8;
		const int TN = 8;

		dim3 blockDim((BN*BM)/(TN*TM));
		dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
		sgemm_vectorized_2D_blocktiling<BM, BN, BK, TM, TN><<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
		cudaDeviceSynchronize();

	}

	else if (version == "sgemm_vectorized_2d_tiled_safe"){
		
		const int BM = 128;
		const int BN = 128;
		const int BK = 8;
		const int TM = 8;
		const int TN = 8;

		dim3 blockDim((BN*BM)/(TN*TM));
		dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
		sgemm_vectorized_2D_blocktiling_safe<BM, BN, BK, TM, TN><<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
		cudaDeviceSynchronize();

	}

	else if (version == "sgemm_vectorized_double_buffering"){

		const int BM = 128;
		const int BN = 128;
		const int BK = 8;
		const int TM = 8;
		const int TN = 8;

		dim3 blockDim((BN*BM)/(TN*TM));
		dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
		sgemm_vectorized_double_buffering<BM, BN, BK, TM, TN><<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
		cudaDeviceSynchronize();
	}
		
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
