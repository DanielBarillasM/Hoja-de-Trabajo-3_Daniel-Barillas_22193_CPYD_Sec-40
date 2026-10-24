#include <cstdio>
#include <cuda_runtime.h>

// Solucion completada a partir de la plantilla vacio.cu proporcionada.

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

    for (int i = 0; i < N; ++i) {
        h_entrada[i] = i * 5;
    }

    int *d_entrada = nullptr;
    int *d_salida = nullptr;
    cudaError_t err;

    err = cudaMalloc((void**)&d_entrada, N * sizeof(int));
    if (err != cudaSuccess) {
        std::printf("Error reservando d_entrada: %s\n", cudaGetErrorString(err));
        delete[] h_entrada;
        delete[] h_salida;
        return -1;
    }

    err = cudaMalloc((void**)&d_salida, N * sizeof(int));
    if (err != cudaSuccess) {
        std::printf("Error reservando d_salida: %s\n", cudaGetErrorString(err));
        cudaFree(d_entrada);
        delete[] h_entrada;
        delete[] h_salida;
        return -1;
    }

    err = cudaMemcpy(d_entrada, h_entrada, N * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::printf("Error copiando entrada al device: %s\n", cudaGetErrorString(err));
        cudaFree(d_entrada);
        cudaFree(d_salida);
        delete[] h_entrada;
        delete[] h_salida;
        return -1;
    }

    int blockSize = 128;
    int gridSize = (N + blockSize - 1) / blockSize;

    clasificarUmbral<<<gridSize, blockSize>>>(d_entrada, d_salida, N, UMBRAL);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::printf("Error al lanzar kernel: %s\n", cudaGetErrorString(err));
        cudaFree(d_entrada);
        cudaFree(d_salida);
        delete[] h_entrada;
        delete[] h_salida;
        return -1;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::printf("Error durante la ejecucion del kernel: %s\n", cudaGetErrorString(err));
        cudaFree(d_entrada);
        cudaFree(d_salida);
        delete[] h_entrada;
        delete[] h_salida;
        return -1;
    }

    err = cudaMemcpy(h_salida, d_salida, N * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::printf("Error copiando salida al host: %s\n", cudaGetErrorString(err));
        cudaFree(d_entrada);
        cudaFree(d_salida);
        delete[] h_entrada;
        delete[] h_salida;
        return -1;
    }

    std::printf("Umbral = %d\n", UMBRAL);
    for (int i = 0; i < N; ++i) {
        std::printf("in[%2d] = %3d  -> out[%2d] = %d\n", i, h_entrada[i], i, h_salida[i]);
    }

    std::printf("Finalizado (host).\n");

    cudaFree(d_entrada);
    cudaFree(d_salida);
    delete[] h_entrada;
    delete[] h_salida;

    return 0;
}
