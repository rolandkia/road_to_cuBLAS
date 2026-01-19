#include <iostream>
#include <string>
#include <unistd.h> 
#include "gemm_kernels.h"
#include "test.h"
#include "benchmark.h"
#include <set>
#include <algorithm>


const std::set<std::string> valid_methods = {
    "sgemm_naive", 
    "sgemm_coalescing",
	"sgemm_tiled",
	"sgemm_1d_tiled",
	"sgemm_2d_tiled",
	"sgemm_vectorized_2d_tiled_safe",
	"sgemm_vectorized_2d_tiled", 
	"sgemm_vectorized_double_buffering"
};


void print_usage(char* prog_name) {
    std::cout << "Usage: " << prog_name << " -m <method> -s <size> [-t]\n"
              << "Options:\n"
              << "  -m : Kernel method parmi : \n";
    for (const auto& m : valid_methods) std::cout <<"\t"+ m << " \n";
    std::cout << "\n  -s : Taille des matrices (N x N)\n"
              << "  -t : Lancer la vérification (test) avant le bench\n";
}

int main(int argc, char** argv) {
    std::string method = "";
    int size = 2048;
    int opt;
	int run_test = 0;

    while ((opt = getopt(argc, argv, "m:s:th")) != -1) {
        switch (opt) {
            case 'm':
                method = optarg;
                break;
            case 's':
                size = std::stoi(optarg);
                break;
            case 't':
                run_test = 1;
                break;
            case 'h':
                print_usage(argv[0]);
                return 0;
            default:
                print_usage(argv[0]);
                return 1;
        }
    }

    if (method.empty()) {
        std::cerr << "Erreur: La méthode (-m) est obligatoire.\n";
        print_usage(argv[0]);
        return 1;
    }


	if (valid_methods.find(method) == valid_methods.end()) {
		std::cerr << "Erreur: Méthode '" << method << "' inconnue.\n";
		std::cerr << "Méthodes disponibles: ";
		for (const auto& m : valid_methods) std::cerr << m << " ";
		std::cerr << std::endl;
		return 1;
	}

    std::cout << "--- Configuration ---" << std::endl;
    std::cout << "Kernel: " << method <<  std::endl;

	if (run_test){
		run_unit_tests(method);
	}
	else{
		run_performance_benchmark(size, size, size, method);
	}
	
    return 0;
}