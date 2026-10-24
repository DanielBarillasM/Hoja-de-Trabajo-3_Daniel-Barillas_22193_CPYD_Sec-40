#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define ANSI_RESET   "\033[0m"
#define ANSI_BOLD    "\033[1m"
#define ANSI_CYAN    "\033[36m"
#define ANSI_GREEN   "\033[32m"
#define ANSI_YELLOW  "\033[33m"
#define ANSI_RED     "\033[31m"
#define ANSI_MAGENTA "\033[35m"

// Kernel para sumar elementos de dos arreglos en paralelo.
// Cada hilo trabaja sobre una posición específica del vector de salida.
__global__ void sumarArreglos(const int *A, const int *B, int *C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    // ============================================================
    // 1) Definición del problema
    // ============================================================
    const int N = 16;

    // ============================================================
    // 2) Declaración e inicialización de arreglos en el host
    // ============================================================
    int *h_A = new int[N];
    int *h_B = new int[N];
    int *h_C = new int[N];

    for (int i = 0; i < N; ++i) {
        h_A[i] = i + 1;
        h_B[i] = 2 * i + 3;
    }

    // ============================================================
    // 3) Declaración de punteros para memoria en el device
    // ============================================================
    int *d_A = nullptr;
    int *d_B = nullptr;
    int *d_C = nullptr;

    cudaError_t err;

    std::printf(ANSI_BOLD ANSI_CYAN "\n========================================\n" ANSI_RESET);
    std::printf(ANSI_BOLD ANSI_CYAN "=== PARTE 1: SUMA DE ARREGLOS EN CUDA ===\n" ANSI_RESET);
    std::printf(ANSI_BOLD ANSI_CYAN "========================================\n\n" ANSI_RESET);

    // ============================================================
    // 4) Reserva de memoria en GPU
    // ============================================================
    err = cudaMalloc((void**)&d_A, N * sizeof(int));
    if (err != cudaSuccess) {
        std::printf(ANSI_RED "[ERROR] No se pudo reservar memoria para d_A: %s\n" ANSI_RESET, cudaGetErrorString(err));
        delete[] h_A;
        delete[] h_B;
        delete[] h_C;
        return -1;
    }

    err = cudaMalloc((void**)&d_B, N * sizeof(int));
    if (err != cudaSuccess) {
        std::printf(ANSI_RED "[ERROR] No se pudo reservar memoria para d_B: %s\n" ANSI_RESET, cudaGetErrorString(err));
        cudaFree(d_A);
        delete[] h_A;
        delete[] h_B;
        delete[] h_C;
        return -1;
    }

    err = cudaMalloc((void**)&d_C, N * sizeof(int));
    if (err != cudaSuccess) {
        std::printf(ANSI_RED "[ERROR] No se pudo reservar memoria para d_C: %s\n" ANSI_RESET, cudaGetErrorString(err));
        cudaFree(d_A);
        cudaFree(d_B);
        delete[] h_A;
        delete[] h_B;
        delete[] h_C;
        return -1;
    }

    std::printf(ANSI_GREEN "[OK] Memoria reservada correctamente en el device.\n" ANSI_RESET);

    // ============================================================
    // 5) Transferencia de datos del host al device
    // ============================================================
    err = cudaMemcpy(d_A, h_A, N * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::printf(ANSI_RED "[ERROR] Fallo al copiar A al device: %s\n" ANSI_RESET, cudaGetErrorString(err));
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        delete[] h_A;
        delete[] h_B;
        delete[] h_C;
        return -1;
    }

    err = cudaMemcpy(d_B, h_B, N * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::printf(ANSI_RED "[ERROR] Fallo al copiar B al device: %s\n" ANSI_RESET, cudaGetErrorString(err));
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        delete[] h_A;
        delete[] h_B;
        delete[] h_C;
        return -1;
    }

    std::printf(ANSI_GREEN "[OK] Datos transferidos del host al device.\n" ANSI_RESET);

    // ============================================================
    // 6) Configuración del kernel
    // ============================================================
    int blockSize = 8;
    int gridSize = (N + blockSize - 1) / blockSize;

    std::printf(ANSI_YELLOW "[INFO] Configuración del kernel: grid = %d, block = %d\n" ANSI_RESET, gridSize, blockSize);

    // ============================================================
    // 7) Ejecución del kernel
    // ============================================================
    sumarArreglos<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::printf(ANSI_RED "[ERROR] Fallo al lanzar el kernel: %s\n" ANSI_RESET, cudaGetErrorString(err));
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        delete[] h_A;
        delete[] h_B;
        delete[] h_C;
        return -1;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::printf(ANSI_RED "[ERROR] El kernel fallo durante la ejecucion: %s\n" ANSI_RESET, cudaGetErrorString(err));
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        delete[] h_A;
        delete[] h_B;
        delete[] h_C;
        return -1;
    }
    std::printf(ANSI_GREEN "[OK] El kernel terminó correctamente.\n" ANSI_RESET);

    // ============================================================
    // 8) Regreso del resultado del device al host
    // ============================================================
    err = cudaMemcpy(h_C, d_C, N * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::printf(ANSI_RED "[ERROR] Fallo al copiar C al host: %s\n" ANSI_RESET, cudaGetErrorString(err));
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        delete[] h_A;
        delete[] h_B;
        delete[] h_C;
        return -1;
    }

    std::printf(ANSI_BOLD ANSI_MAGENTA "\n=== Arreglo A ===\n" ANSI_RESET);
    for (int i = 0; i < N; ++i) {
        std::printf("A[%2d] = %3d\n", i, h_A[i]);
    }

    std::printf(ANSI_BOLD ANSI_MAGENTA "\n=== Arreglo B ===\n" ANSI_RESET);
    for (int i = 0; i < N; ++i) {
        std::printf("B[%2d] = %3d\n", i, h_B[i]);
    }

    std::printf(ANSI_BOLD ANSI_GREEN "\n=== Resultado A + B ===\n" ANSI_RESET);
    for (int i = 0; i < N; ++i) {
        std::printf("C[%2d] = %3d\n", i, h_C[i]);
    }

    // ============================================================
    // 9) Liberación de memoria
    // ============================================================
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    std::printf(ANSI_BOLD ANSI_GREEN "\n✅ Programa finalizado correctamente.\n" ANSI_RESET);
    return 0;
}
