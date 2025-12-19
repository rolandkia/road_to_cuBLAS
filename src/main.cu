#include <iostream>
#include <string>
#include "gemm_kernels.h"
#include "test.h"
#include "benchmark.h"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Usage: ./gemm_tool [test | bench]" << std::endl;
        return 1;
    }

    std::string mode = argv[1];

    if (mode == "test") {
        std::cout << "--- Mode Validation ---" << std::endl;
        run_unit_tests(); 
    } 
    else if (mode == "bench") {
        std::cout << "--- Mode Benchmark ---" << std::endl;
        // On peut passer des tailles en arguments : ./gemm_tool bench 2048
        int size = (argc > 2) ? std::stoi(argv[2]) : 2048;
        run_performance_benchmark(size, size, size); 
    } 
    else {
        std::cerr << "Mode inconnu : " << mode << std::endl;
        return 1;
    }

    return 0;
}