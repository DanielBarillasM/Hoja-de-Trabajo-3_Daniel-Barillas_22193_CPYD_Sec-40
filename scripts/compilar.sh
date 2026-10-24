#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."

mkdir -p build results

printf "== Compilando Parte 1: Suma CUDA ==\n"
nvcc -std=c++17 -O2 source/parte1_suma_cuda.cu -o build/parte1_suma_cuda
./build/parte1_suma_cuda | tee results/parte1_suma_cuda.txt

printf "\n== Compilando Parte 2: Umbral CUDA ==\n"
nvcc -std=c++17 -O2 source/parte2_umbral_clasificacion.cu -o build/parte2_umbral_clasificacion
./build/parte2_umbral_clasificacion | tee results/parte2_umbral_clasificacion.txt

printf "\n== Compilando Parte 3: CPU ==\n"
g++ -std=c++17 -O2 source/parte3_cpu_vs_gpu.cpp -o build/parte3_cpu_vs_gpu
./build/parte3_cpu_vs_gpu | tee results/parte3_cpu_vs_gpu.txt

printf "\nTodos los programas fueron compilados y ejecutados correctamente.\n"
