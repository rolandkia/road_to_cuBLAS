#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include "gemm_kernels.h"
#include "test.h"
#include <string.h>

// Fonction de référence sur CPU (Algorithme naïf O(N^3))
void cpu_gemm_reference(float* A, float* B, float* C, int M, int K, int N) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0;
            for (int l = 0; l < K; ++l) {
                sum += A[i * K + l] * B[l * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

bool verify_results(float* gpu_res, float* cpu_res, int M, int N) {
   
    float epsilon = 1e-3f; 
    
    for (int i = 0; i < M * N; ++i) {
        float abs_err = std::abs(gpu_res[i] - cpu_res[i]);
        float rel_err = abs_err / (std::abs(cpu_res[i]) + 1e-9f);

        if (rel_err > epsilon) {
            printf("Erreur à l'index %d :\n", i);
            printf("  Ma version: %f\n  Réference: %f\n", gpu_res[i], cpu_res[i]);
            printf("  Erreur relative: %e\n", rel_err);
            return false;
        }
    }
    return true;
}

void print_matrix(float * matrix, int M, int N){
	printf("\n");

	for (int i = 0; i<M; i++){
		for (int j = 0; j<N; j++){
			printf("%f ", matrix[i*N + j]);
		}
		printf("\n");
	}
	printf("\n");
}


void run_single_test(int M, int K, int N, int cublas, std::string version) {
	
	if (!cublas){
		printf("Test GEMM [%d x %d x %d] :", M, K, N);

	}
	else{
    	printf("Test cuBLAS GEMM [%d x %d x %d] :", M, K, N);
	}

    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

	float * h_A = (float *)malloc(size_A);
	float * h_B = (float *)malloc(size_B);
	float * h_C_gpu = (float *)malloc(size_C);
	float * h_C_cpu = (float *)malloc(size_C);

    for (int i = 0; i < M * K; ++i) h_A[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < K * N; ++i) h_B[i] = (float)rand() / RAND_MAX;

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, size_C); 

	if (cublas){
		const float alpha = 1.0;
    	const float beta = 0.0;
		
		cublasHandle_t handle;
    	cublasCreate(&handle);

		cublas_sgemm(handle, d_A, d_B, d_C, alpha, beta, M, N, K);

		cudaError_t err = cudaGetLastError();
		if (err != cudaSuccess) {
			printf("CUDA Error: %s\n", cudaGetErrorString(err));
		}

	}
	else{
		sgemm_cuda(d_A, d_B, d_C, M, K, N, version);

		cudaError_t err = cudaGetLastError();
		if (err != cudaSuccess) {
			printf("CUDA Error: %s\n", cudaGetErrorString(err));
		}
	}

    cudaMemcpy(h_C_gpu, d_C, size_C, cudaMemcpyDeviceToHost);

    cpu_gemm_reference(h_A, h_B, h_C_cpu, M, K, N);

    if (verify_results(h_C_gpu, h_C_cpu, M, N)) {
        printf("PASS\n");
    } else {
        printf("FAIL\n");
    }

	if (M < 2 && N < 2){
		print_matrix(h_C_cpu, M, N);
		print_matrix(h_C_gpu, M, N);
	}

	free(h_A);
	free(h_B);
	free(h_C_gpu);
	free(h_C_cpu);
    cudaFree(d_A); 
	cudaFree(d_B); 
	cudaFree(d_C);
}

void run_unit_tests(std::string version) {

	int gen_test = 1;
	if ((version == "sgemm_vectorized_double_buffering") || (version == "sgemm_vectorized_2d_tiled")){
		gen_test = 0;
	}

	run_single_test(4, 4, 4, 0, version);
	run_single_test(4, 4, 4, 1, version);

    // Test 1 : Petit cas simple (multiple de la taille des blocs)
    run_single_test(32, 32, 32, 0, version);
	run_single_test(32, 32, 32, 1, version);


    // Test 2 : Matrices rectangulaires
    run_single_test(64, 128, 64, 0, version);
    run_single_test(64, 128, 64, 1, version);


	if (gen_test){
		// Test 3 : Tailles impaires (Vérifie si vous gérez bien les bords "out of bounds")
		run_single_test(33, 33, 33, 0, version);
		run_single_test(100, 50, 150, 0, version);

		run_single_test(33, 33, 33, 1, version);
		run_single_test(100, 50, 150, 1, version);
	}

	// Test 4 : Grosse Tailles
	run_single_test(1024, 1024, 1024, 1, version);
	run_single_test(1024, 1024, 1024, 0, version);

	// run_single_test(2048, 2048, 2048, 1, version);
	// run_single_test(2048, 2048, 2048, 0, version);

}