#include "benchmark.h"
#include "gemm_kernels.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>


#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA Error: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
        exit(1); \
    } \
}

double calculate_gflops(int M, int K, int N, float milliseconds) {
    double ops = 2.0 * (double)M * (double)N * (double)K;
    return (ops * 1e-6) / (double)milliseconds; 
}

float perf_dgemm(double* d_A, double* d_B, double* d_C, int M, int K, int N, int iterations) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warm-up
    dgemm_cuda(d_A, d_B, d_C, M, K, N);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for(int i = 0; i < iterations; i++) {
        dgemm_cuda(d_A, d_B, d_C, M, K, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms / iterations;
}


float perf_dgemm_cublas(cublasHandle_t handle, double* d_A, double* d_B, double* d_C, int M, int K, int N, int iterations) {
    const double alpha = 1.0;
    const double beta = 0.0;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warm-up
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for(int i = 0; i < iterations; i++) {
        cublasStatus_t status = cublasDgemm(handle, 
                                            CUBLAS_OP_N, CUBLAS_OP_N, 
                                            N, M, K, 
                                            &alpha, 
                                            d_B, N,  // ldb
                                            d_A, K,  // lda
                                            &beta, 
                                            d_C, N); // ldc
        if (status != CUBLAS_STATUS_SUCCESS) {
            std::cerr << "Erreur cuBLAS !" << std::endl;
            return -1.0f;
        }
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms / iterations;
}


void run_performance_benchmark(int M, int K, int N) {
    int iterations = 10;
    const double alpha = 1.0;
    const double beta = 0.0;

    printf("\n=== Benchmark GEMM: M=%d, K=%d, N=%d ===\n", M, K, N);

    // 1. Allocation et Initialisation (C++ côté CPU)
    size_t size_A = M * K * sizeof(double);
    size_t size_B = K * N * sizeof(double);
    size_t size_C = M * N * sizeof(double);

    double *h_A = (double*)malloc(size_A);
    double *h_B = (double*)malloc(size_B);
    for(int i = 0; i < M*K; i++) h_A[i] = 1.0; // Remplissage avec des vraies valeurs
    for(int i = 0; i < K*N; i++) h_B[i] = 0.01;

    // 2. Préparation Device
    double *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, size_A));
    CHECK_CUDA(cudaMalloc(&d_B, size_B));
    CHECK_CUDA(cudaMalloc(&d_C, size_C));

    CHECK_CUDA(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));


	float my_perf = perf_dgemm(d_A, d_B, d_C, M, K, N, iterations);
	float cublas_perf = perf_dgemm_cublas(d_A, d_B, d_C, M, K, N, iterations);

    double my_gflops = calculate_gflops(M, K, N, my_perf);
    double cublas_gflops = calculate_gflops(M, K, N, cublas_perf);

    std::cout << std::fixed << std::setprecision(3);
    std::cout << ">>> Votre Kernel : " << custom_ms << " ms | " << my_gflops << " GFLOPS" << std::endl;
    std::cout << ">>> cuBLAS       : " << cublas_ms << " ms | " << cublas_gflops << " GFLOPS" << std::endl;
    std::cout << ">>> Efficacité   : " << (my_gflops / cublas_gflops) * 100.0 << "% de cuBLAS" << std::endl;

    // // Nettoyage
    free(h_A); free(h_B);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cublasDestroy(handle);
    cudaEventDestroy(start); 
	cudaEventDestroy(stop);
}