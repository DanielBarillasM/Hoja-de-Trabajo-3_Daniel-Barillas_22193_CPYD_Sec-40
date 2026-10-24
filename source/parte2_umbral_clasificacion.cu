#include <cstdio>
#include <cuda_runtime.h>

#define ANSI_RESET   "\033[0m"
#define ANSI_BOLD    "\033[1m"
#define ANSI_CYAN    "\033[36m"
#define ANSI_GREEN   "\033[32m"
#define ANSI_YELLOW  "\033[33m"
#define ANSI_RED     "\033[31m"
#define ANSI_MAGENTA "\033[35m"

// El kernel recibe un arreglo de entrada y devuelve una salida binaria.
// Si el valor es mayor o igual que el umbral, la salida es 1; de lo contrario, 0.
__global__ void clasificarUmbral(const int *entrada, int *salida, int N, int umbral) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        salida[idx] = (entrada[idx] >= umbral) ? 1 : 0;
    }
}

int main() {
    const int N = 32;
    const int UMBRAL = 50;

    int *h_entrada = new int[N];
    int *h_salida = new int[N];

    // Inicialización del arreglo de entrada.
    for (int i = 0; i < N; ++i) {
        h_entrada[i] = i * 5;
    }

    int *d_entrada = nullptr;
    int *d_salida = nullptr;
    cudaError_t err;

    std::printf(ANSI_BOLD ANSI_CYAN "\n=========================================\n" ANSI_RESET);
    std::printf(ANSI_BOLD ANSI_CYAN "=== PARTE 2: CATEGORIZACIÓN POR UMBRAL ===\n" ANSI_RESET);
    std::printf(ANSI_BOLD ANSI_CYAN "=========================================\n\n" ANSI_RESET);

    // Reserva de memoria en GPU.
    err = cudaMalloc((void**)&d_entrada, N * sizeof(int));
    if (err != cudaSuccess) {
        std::printf(ANSI_RED "[ERROR] No se pudo reservar d_entrada: %s\n" ANSI_RESET, cudaGetErrorString(err));
        delete[] h_entrada;
        delete[] h_salida;
        return -1;
    }

    err = cudaMalloc((void**)&d_salida, N * sizeof(int));
    if (err != cudaSuccess) {
        std::printf(ANSI_RED "[ERROR] No se pudo reservar d_salida: %s\n" ANSI_RESET, cudaGetErrorString(err));
        cudaFree(d_entrada);
        delete[] h_entrada;
        delete[] h_salida;
        return -1;
    }

    std::printf(ANSI_GREEN "[OK] Memoria reservada correctamente.\n" ANSI_RESET);

    // Copia del arreglo desde el host al device.
    err = cudaMemcpy(d_entrada, h_entrada, N * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::printf(ANSI_RED "[ERROR] Fallo al copiar entrada al device: %s\n" ANSI_RESET, cudaGetErrorString(err));
        cudaFree(d_entrada);
        cudaFree(d_salida);
        delete[] h_entrada;
        delete[] h_salida;
        return -1;
    }

    std::printf(ANSI_GREEN "[OK] Datos transferidos al GPU.\n" ANSI_RESET);

    int blockSize = 128;
    int gridSize = (N + blockSize - 1) / blockSize;
    std::printf(ANSI_YELLOW "[INFO] Configuración del kernel: grid = %d, block = %d\n" ANSI_RESET, gridSize, blockSize);

    // Lanzamiento del kernel.
    clasificarUmbral<<<gridSize, blockSize>>>(d_entrada, d_salida, N, UMBRAL);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::printf(ANSI_RED "[ERROR] Fallo al lanzar el kernel: %s\n" ANSI_RESET, cudaGetErrorString(err));
        cudaFree(d_entrada);
        cudaFree(d_salida);
        delete[] h_entrada;
        delete[] h_salida;
        return -1;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::printf(ANSI_RED "[ERROR] El kernel fallo durante la ejecucion: %s\n" ANSI_RESET, cudaGetErrorString(err));
        cudaFree(d_entrada);
        cudaFree(d_salida);
        delete[] h_entrada;
        delete[] h_salida;
        return -1;
    }
    std::printf(ANSI_GREEN "[OK] El kernel completó la clasificación.\n" ANSI_RESET);

    // Copia del resultado del device al host.
    err = cudaMemcpy(h_salida, d_salida, N * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::printf(ANSI_RED "[ERROR] Fallo al traer la salida al host: %s\n" ANSI_RESET, cudaGetErrorString(err));
        cudaFree(d_entrada);
        cudaFree(d_salida);
        delete[] h_entrada;
        delete[] h_salida;
        return -1;
    }

    std::printf(ANSI_BOLD ANSI_MAGENTA "\n=== Resultado de clasificación por umbral ===\n" ANSI_RESET);
    std::printf("Umbral = %d\n", UMBRAL);
    for (int i = 0; i < N; ++i) {
        std::printf("in[%2d] = %3d  -> out[%2d] = %d\n", i, h_entrada[i], i, h_salida[i]);
    }

    std::printf(ANSI_BOLD ANSI_GREEN "\n✅ Finalizado (host).\n" ANSI_RESET);

    cudaFree(d_entrada);
    cudaFree(d_salida);
    delete[] h_entrada;
    delete[] h_salida;

    return 0;
}
