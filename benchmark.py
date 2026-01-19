import subprocess
import re
import matplotlib.pyplot as plt
import numpy as np

BINARY_PATH = "./build/gemm_tool"
METHODS = [
    "sgemm_naive", 
    "sgemm_coalescing",
	"sgemm_tiled",
	"sgemm_1d_tiled",
	"sgemm_2d_tiled",
	"sgemm_vectorized_2d_tiled_safe",
	"sgemm_vectorized_2d_tiled", 
	"sgemm_vectorized_double_buffering", 
	"cuBLAS_sgemm"
]

SIZES = [2**i for i in range(8, 13)] 

results = {method: [] for method in METHODS}

def run_benchmark(method, size):
    
    
    if (method == "cuBLAS_sgemm"):
        cmd = [BINARY_PATH, "-m", METHODS[-2], "-s", str(size)]
    else:
        cmd = [BINARY_PATH, "-m", method, "-s", str(size)]

    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate()
        
        if method == "cuBLAS_sgemm":
            match = re.search(r">>> cuBLAS\s+:\s+[\d.]+\s+ms\s+\|\s+([\d.]+)\s+GFLOPS", stdout)
        else:
            match = re.search(r">>> Kernel \(.*?\) :\s+[\d.]+\s+ms\s+\|\s+([\d.]+)\s+GFLOPS", stdout)
            
        if match:
            return float(match.group(1))
        return 0.0
    except Exception as e:
        print(f"Erreur sur {method} à taille {size}: {e}")
        return 0.0

for size in SIZES:
    print(f"Benchmarking Size: {size}x{size}...")
    for method in METHODS:
        gflops = run_benchmark(method, size)
        results[method].append(gflops)
        print(f"  {method}: {gflops} GFLOPS")

plt.figure(figsize=(12, 7))
for method, gflops_list in results.items():
    plt.plot(SIZES, gflops_list, marker='o', label=method)

plt.xlabel("Matrix Size (N x N)")
plt.ylabel("Performance (GFLOPS)")
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.legend()
plt.savefig("sgemm_performance.png")
plt.show()

print("\nGraphique sauvegardé sous 'sgemm_performance.png'")