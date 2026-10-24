#include <cstdio>
#include <cuda_runtime.h>

__global__ void umbralizar(const int *in, int *out, int N, int umbral) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        out[i] = (in[i] >= umbral) ? 1 : 0;
    }
}

int main() {
    const int N = 32;
    const int UMBRAL = 50;

    int *h_in = new int[N];
    int *h_out = new int[N];

    for (int i = 0; i < N; ++i) {
        h_in[i] = i * 5;
    }

    int *d_in = nullptr;
    int *d_out = nullptr;

    cudaError_t err;

    err = cudaMalloc((void**)&d_in, N * sizeof(int));
    if (err != cudaSuccess) {
        std::printf("Error en cudaMalloc(d_in): %s\n", cudaGetErrorString(err));
        delete[] h_in;
        delete[] h_out;
        return -1;
    }

    err = cudaMalloc((void**)&d_out, N * sizeof(int));
    if (err != cudaSuccess) {
        std::printf("Error en cudaMalloc(d_out): %s\n", cudaGetErrorString(err));
        cudaFree(d_in);
        delete[] h_in;
        delete[] h_out;
        return -1;
    }

    err = cudaMemcpy(d_in, h_in, N * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::printf("Error en cudaMemcpy Host->Device: %s\n", cudaGetErrorString(err));
        cudaFree(d_in);
        cudaFree(d_out);
        delete[] h_in;
        delete[] h_out;
        return -1;
    }

    int blockSize = 128;
    int gridSize = (N + blockSize - 1) / blockSize;

    umbralizar<<<gridSize, blockSize>>>(d_in, d_out, N, UMBRAL);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::printf("Error al lanzar kernel: %s\n", cudaGetErrorString(err));
        cudaFree(d_in);
        cudaFree(d_out);
        delete[] h_in;
        delete[] h_out;
        return -1;
    }

    cudaDeviceSynchronize();

    err = cudaMemcpy(h_out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::printf("Error en cudaMemcpy Device->Host: %s\n", cudaGetErrorString(err));
        cudaFree(d_in);
        cudaFree(d_out);
        delete[] h_in;
        delete[] h_out;
        return -1;
    }

    printf("Umbral = %d\n", UMBRAL);
    for (int i = 0; i < N; ++i) {
        printf("in[%2d] = %3d  -> out[%2d] = %d\n", i, h_in[i], i, h_out[i]);
    }

    cudaFree(d_in);
    cudaFree(d_out);

    delete[] h_in;
    delete[] h_out;

    printf("Finalizado (host).\n");
    return 0;
}
