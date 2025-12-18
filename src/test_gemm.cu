#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include "gemm_kernels.h"
#include "test.h"

// Fonction de référence sur CPU (Algorithme naïf O(N^3))
void cpu_gemm_reference(double* A, double* B, double* C, int M, int K, int N) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            double sum = 0.0;
            for (int l = 0; l < K; ++l) {
                sum += A[i * K + l] * B[l * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// Vérification de la différence entre deux matrices
bool verify_results(double* gpu_res, double* cpu_res, int M, int N) {
    double epsilon = 1e-8; 
    for (int i = 0; i < M * N; ++i) {
        if (std::abs(gpu_res[i] - cpu_res[i]) > epsilon) {
            printf("Erreur d'exactitude à l'index %d : GPU=%f, CPU=%f\n", i, gpu_res[i], cpu_res[i]);
            return false;
        }
    }
    return true;
}

void run_single_test(int M, int K, int N) {
    printf("Test GEMM [%d x %d x %d] : ", M, K, N);

    size_t size_A = M * K * sizeof(double);
    size_t size_B = K * N * sizeof(double);
    size_t size_C = M * N * sizeof(double);

    std::vector<double> h_A(M * K), h_B(K * N), h_C_gpu(M * N), h_C_cpu(M * N);

    for (int i = 0; i < M * K; ++i) h_A[i] = (double)rand() / RAND_MAX;
    for (int i = 0; i < K * N; ++i) h_B[i] = (double)rand() / RAND_MAX;

    double *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    cudaMemcpy(d_A, h_A.data(), size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), size_B, cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, size_C); 

    dgemm_cuda(d_A, d_B, d_C, M, K, N);

    cudaMemcpy(h_C_gpu.data(), d_C, size_C, cudaMemcpyDeviceToHost);

    cpu_gemm_reference(h_A.data(), h_B.data(), h_C_cpu.data(), M, K, N);

    if (verify_results(h_C_gpu.data(), h_C_cpu.data(), M, N)) {
        printf("PASS\n");
    } else {
        printf("FAIL\n");
    }

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}

void run_unit_tests() {
    // Test 1 : Petit cas simple (multiple de la taille des blocs)
    run_single_test(32, 32, 32);

    // Test 2 : Matrices rectangulaires
    run_single_test(64, 128, 64);

    // Test 3 : Tailles impaires (Vérifie si vous gérez bien les bords "out of bounds")
    run_single_test(33, 33, 33);
    run_single_test(100, 50, 150);

    printf("\nTous les tests sont terminés.\n");
}