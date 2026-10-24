#!/usr/bin/env bash
# Este archivo debe conservar finales de linea LF para ejecutarse en Bash/WSL.
set -euo pipefail

cd "$(dirname "$0")/.."

mkdir -p build results

command -v nvcc >/dev/null 2>&1 || {
    printf "ERROR: nvcc no esta disponible en PATH.\n" >&2
    exit 1
}

command -v g++ >/dev/null 2>&1 || {
    printf "ERROR: g++ no esta disponible en PATH.\n" >&2
    exit 1
}

printf "== Compilando Parte 1: Suma CUDA ==\n"
nvcc -std=c++17 -O2 source/parte1_suma_cuda.cu -o build/parte1_suma_cuda
./build/parte1_suma_cuda | tee results/parte1_suma_cuda.txt

printf "\n== Compilando Parte 2: Umbral CUDA ==\n"
nvcc -std=c++17 -O2 source/parte2_umbral_clasificacion.cu -o build/parte2_umbral_clasificacion
./build/parte2_umbral_clasificacion | tee results/parte2_umbral_clasificacion.txt

printf "\n== Compilando Parte 2: plantilla vacio.cu completada ==\n"
nvcc -std=c++17 -O2 source/vacio.cu -o build/vacio
./build/vacio | tee results/vacio.txt

printf "\n== Compilando Parte 3: version CPU ==\n"
g++ -std=c++17 -O2 source/parte3_cpu_vs_gpu.cpp -o build/parte3_cpu_vs_gpu
./build/parte3_cpu_vs_gpu | tee results/parte3_cpu_vs_gpu.txt

printf "\n== Compilando Parte 3: comparacion CPU vs GPU ==\n"
nvcc -std=c++17 -O2 source/parte3_comparacion_cpu_gpu.cu -o build/parte3_comparacion_cpu_gpu
./build/parte3_comparacion_cpu_gpu | tee results/parte3_comparacion_cpu_gpu.txt

printf "\nTodos los programas fueron compilados y ejecutados correctamente.\n"
