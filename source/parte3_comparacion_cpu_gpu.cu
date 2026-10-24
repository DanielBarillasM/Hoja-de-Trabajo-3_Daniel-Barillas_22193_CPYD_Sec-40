#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                               \
    do {                                                                               \
        cudaError_t status_ = (call);                                                   \
        if (status_ != cudaSuccess) {                                                   \
            std::fprintf(stderr, "[ERROR CUDA] %s:%d: %s\n", __FILE__, __LINE__,      \
                         cudaGetErrorString(status_));                                  \
            return EXIT_FAILURE;                                                       \
        }                                                                              \
    } while (0)

__global__ void sumarArreglosGPU(const int *A, const int *B, int *C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

void sumarArreglosCPU(const std::vector<int>& A,
                      const std::vector<int>& B,
                      std::vector<int>& C) {
    for (std::size_t i = 0; i < A.size(); ++i) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    constexpr int N = 8 * 1024 * 1024;
    constexpr int BLOCK_SIZE = 256;
    const int gridSize = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const std::size_t bytes = static_cast<std::size_t>(N) * sizeof(int);

    std::vector<int> h_A(N);
    std::vector<int> h_B(N);
    std::vector<int> h_C_cpu(N);
    std::vector<int> h_C_gpu(N);

    for (int i = 0; i < N; ++i) {
        h_A[i] = i + 1;
        h_B[i] = 2 * i + 3;
    }

    std::printf("============================================================\n");
    std::printf(" PARTE 3: COMPARACION JUSTA CPU VS GPU\n");
    std::printf("============================================================\n");
    std::printf("Elementos: %d\n", N);
    std::printf("Datos por arreglo: %.2f MiB\n", bytes / (1024.0 * 1024.0));
    std::printf("Configuracion CUDA: grid=%d, block=%d\n\n", gridSize, BLOCK_SIZE);

    const auto inicioCPU = std::chrono::steady_clock::now();
    sumarArreglosCPU(h_A, h_B, h_C_cpu);
    const auto finCPU = std::chrono::steady_clock::now();
    const double tiempoCPUms =
        std::chrono::duration<double, std::milli>(finCPU - inicioCPU).count();

    int *d_A = nullptr;
    int *d_B = nullptr;
    int *d_C = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_B), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_C), bytes));

    cudaDeviceProp propiedades{};
    CUDA_CHECK(cudaGetDeviceProperties(&propiedades, 0));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice));

    // Calentamiento: evita que la inicializacion del contexto distorsione la medicion.
    sumarArreglosGPU<<<gridSize, BLOCK_SIZE>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t inicioKernel{};
    cudaEvent_t finKernel{};
    CUDA_CHECK(cudaEventCreate(&inicioKernel));
    CUDA_CHECK(cudaEventCreate(&finKernel));
    CUDA_CHECK(cudaEventRecord(inicioKernel));
    sumarArreglosGPU<<<gridSize, BLOCK_SIZE>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(finKernel));
    CUDA_CHECK(cudaEventSynchronize(finKernel));

    float tiempoKernelMs = 0.0F;
    CUDA_CHECK(cudaEventElapsedTime(&tiempoKernelMs, inicioKernel, finKernel));

    // Tiempo total GPU: incluye H2D, kernel y D2H; excluye cudaMalloc/cudaFree.
    const auto inicioTotalGPU = std::chrono::steady_clock::now();
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice));
    sumarArreglosGPU<<<gridSize, BLOCK_SIZE>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(h_C_gpu.data(), d_C, bytes, cudaMemcpyDeviceToHost));
    const auto finTotalGPU = std::chrono::steady_clock::now();
    const double tiempoTotalGPUms =
        std::chrono::duration<double, std::milli>(finTotalGPU - inicioTotalGPU).count();

    std::size_t errores = 0;
    for (int i = 0; i < N; ++i) {
        const int esperado = h_A[i] + h_B[i];
        if (h_C_cpu[i] != esperado || h_C_gpu[i] != esperado) {
            ++errores;
            if (errores <= 3) {
                std::fprintf(stderr,
                             "[ERROR] i=%d, esperado=%d, CPU=%d, GPU=%d\n",
                             i, esperado, h_C_cpu[i], h_C_gpu[i]);
            }
        }
    }

    const double aceleracionKernel = tiempoCPUms / static_cast<double>(tiempoKernelMs);
    const double aceleracionTotal = tiempoCPUms / tiempoTotalGPUms;

    std::printf("GPU: %s\n", propiedades.name);
    std::printf("Tiempo CPU:              %.3f ms\n", tiempoCPUms);
    std::printf("Tiempo GPU (solo kernel): %.3f ms\n", tiempoKernelMs);
    std::printf("Tiempo GPU (H2D+kernel+D2H): %.3f ms\n", tiempoTotalGPUms);
    std::printf("Aceleracion solo kernel: %.2fx\n", aceleracionKernel);
    std::printf("Aceleracion extremo a extremo: %.2fx\n", aceleracionTotal);
    std::printf("Validacion: %s (%zu discrepancias)\n",
                errores == 0 ? "CORRECTA" : "INCORRECTA", errores);
    std::printf("Primeros resultados: %d, %d, %d, %d, %d\n",
                h_C_gpu[0], h_C_gpu[1], h_C_gpu[2], h_C_gpu[3], h_C_gpu[4]);

    CUDA_CHECK(cudaEventDestroy(inicioKernel));
    CUDA_CHECK(cudaEventDestroy(finKernel));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return errores == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
