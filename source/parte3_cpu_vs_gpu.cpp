#include <chrono>
#include <cstdlib>
#include <iostream>
#include <vector>

#define ANSI_RESET   "\033[0m"
#define ANSI_BOLD    "\033[1m"
#define ANSI_CYAN    "\033[36m"
#define ANSI_GREEN   "\033[32m"
#define ANSI_YELLOW  "\033[33m"
#define ANSI_RED     "\033[31m"
#define ANSI_MAGENTA "\033[35m"

// Función secuencial en CPU: suma elemento a elemento.
std::vector<int> sumaCPU(const std::vector<int>& A, const std::vector<int>& B) {
    std::vector<int> C(A.size());

    for (size_t i = 0; i < A.size(); ++i) {
        C[i] = A[i] + B[i];
    }

    return C;
}

int main() {
    const int N = 1000000;

    std::vector<int> A(N);
    std::vector<int> B(N);

    for (int i = 0; i < N; ++i) {
        A[i] = i + 1;
        B[i] = 2 * i + 3;
    }

    std::cout << ANSI_BOLD ANSI_CYAN << "\n=========================================" << ANSI_RESET << std::endl;
    std::cout << ANSI_BOLD ANSI_CYAN << "=== PARTE 3: VERSION SECUENCIAL EN CPU ===" << ANSI_RESET << std::endl;
    std::cout << ANSI_BOLD ANSI_CYAN << "=========================================" << ANSI_RESET << std::endl;

    std::cout << ANSI_YELLOW << "[INFO] Generando arreglos de tamaño " << N << " ..." << ANSI_RESET << std::endl;

    auto inicioCPU = std::chrono::high_resolution_clock::now();
    std::vector<int> C = sumaCPU(A, B);
    auto finCPU = std::chrono::high_resolution_clock::now();

    auto duracionCPU = std::chrono::duration_cast<std::chrono::microseconds>(finCPU - inicioCPU).count();

    std::cout << ANSI_BOLD ANSI_MAGENTA << "\n=== Primeros 5 resultados ===" << ANSI_RESET << std::endl;
    for (int i = 0; i < 5; ++i) {
        std::cout << "C[" << i << "] = " << C[i] << std::endl;
    }

    std::cout << ANSI_GREEN << "\nTiempo CPU: " << duracionCPU << " microsegundos" << ANSI_RESET << std::endl;
    std::cout << ANSI_GREEN << "✅ La versión CPU fue ejecutada correctamente." << ANSI_RESET << std::endl;

    bool correcto = true;
    for (int i = 0; i < N; ++i) {
        if (C[i] != A[i] + B[i]) {
            correcto = false;
            break;
        }
    }

    std::cout << (correcto ? ANSI_GREEN : ANSI_RED)
              << "[VALIDACION] Resultado CPU: " << (correcto ? "CORRECTO" : "INCORRECTO")
              << ANSI_RESET << std::endl;

    return correcto ? 0 : 1;
}
