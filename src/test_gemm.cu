#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include "gemm_kernels.h"
#include "test.h"

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
    // Augmente l'epsilon pour le float (1e-3 est plus réaliste pour de grandes matrices)
    float epsilon = 1e-3f; 
    
    for (int i = 0; i < M * N; ++i) {
        float abs_err = std::abs(gpu_res[i] - cpu_res[i]);
        float rel_err = abs_err / (std::abs(cpu_res[i]) + 1e-9f);

        if (rel_err > epsilon) {
            printf("Erreur à l'index %d :\n", i);
            printf("  GPU: %f\n  CPU: %f\n", gpu_res[i], cpu_res[i]);
            printf("  Erreur relative: %e\n", rel_err);
            return false;
        }
    }
    printf("Vérification réussie !\n");
    return true;
}

void run_single_test(int M, int K, int N) {
    printf("Test GEMM [%d x %d x %d] : ", M, K, N);

    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    std::vector<float> h_A(M * K), h_B(K * N), h_C_gpu(M * N), h_C_cpu(M * N);

    for (int i = 0; i < M * K; ++i) h_A[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < K * N; ++i) h_B[i] = (float)rand() / RAND_MAX;

    float *d_A, *d_B, *d_C;
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