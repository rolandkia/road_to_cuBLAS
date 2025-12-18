#include "benchmark.h"
#include "gemm_kernels.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>

double calculate_gflops(int M, int K, int N, float milliseconds) {
    double ops = 2.0 * (double)M * (double)N * (double)K;
    return (ops * 1e-6) / (double)milliseconds; 
}

void run_performance_benchmark(int M, int K, int N) {
    printf("\n--- Benchmark GEMM: %d x %d x %d ---\n", M, K, N);

    size_t size_A = M * K * sizeof(double);
    size_t size_B = K * N * sizeof(double);
    size_t size_C = M * N * sizeof(double);

    double *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    cudaMemset(d_A, 1, size_A);
    cudaMemset(d_B, 1, size_B);

    cublasHandle_t handle;
    cublasCreate(&handle);
    const double alpha = 1.0;
    const double beta = 0.0;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warm-up (pour réveiller le GPU)
    dgemm_cuda(d_A, d_B, d_C, M, K, N);
    
    cudaEventRecord(start);
    int iterations = 10;
    for(int i = 0; i < iterations; i++) {
        dgemm_cuda(d_A, d_B, d_C, M, K, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float my_ms = 0;
    cudaEventElapsedTime(&my_ms, start, stop);
    my_ms /= iterations; // Temps moyen

    // cuBLAS est en Column-Major. Pour simuler du Row-Major (C = A*B),
    // on demande à cuBLAS de calculer C^T = B^T * A^T
    cudaEventRecord(start);
    for(int i = 0; i < iterations; i++) {
        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                    N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float cublas_ms = 0;
    cudaEventElapsedTime(&cublas_ms, start, stop);
    cublas_ms /= iterations;

    double my_gflops = calculate_gflops(M, K, N, my_ms);
    double cublas_gflops = calculate_gflops(M, K, N, cublas_ms);

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Votre Kernel : " << my_ms << " ms | " << my_gflops << " GFLOPS" << std::endl;
    std::cout << "cuBLAS       : " << cublas_ms << " ms | " << cublas_gflops << " GFLOPS" << std::endl;
    std::cout << "Efficacité   : " << (my_gflops / cublas_gflops) * 100.0 << "% de cuBLAS" << std::endl;

    cublasDestroy(handle);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}