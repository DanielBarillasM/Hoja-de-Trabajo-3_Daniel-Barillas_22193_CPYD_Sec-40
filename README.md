> # Hoja de Trabajo 3 — Programación Paralela y Distribuida

<p align="center">
  <img src="https://img.shields.io/badge/CUDA-Enabled-76B900?style=for-the-badge&logo=nvidia" alt="CUDA" />
  <img src="https://img.shields.io/badge/WSL-2-0078D4?style=for-the-badge&logo=windows" alt="WSL 2" />
  <img src="https://img.shields.io/badge/Language-C%2B%2B-00599C?style=for-the-badge&logo=c%2B%2B" alt="C++" />
  <img src="https://img.shields.io/badge/Status-Completed-2ECC71?style=for-the-badge" alt="Status" />
</p>

## Información académica
- Universidad: Universidad del Valle de Guatemala
- Curso: Programación Paralela y Distribuida
- Sección: 40
- Estudiante: Daniel Barillas
- Carné: 22193
- Fecha de entrega: Miércoles 21 de octubre
- Tema: CUDA, paralelismo de datos, memorización y análisis de rendimiento

---

## Descripción general
Este proyecto presenta tres actividades fundamentales de programación paralela con CUDA:

1. Suma de dos arreglos en paralelo usando GPU.
2. Clasificación por umbral con un kernel CUDA.
3. Comparación de rendimiento entre CPU y GPU usando medición de tiempo.

La finalidad es comprender cómo se ejecutan operaciones en el host y en el device, cómo se administra la memoria y cómo el paralelismo cambia el tiempo de cómputo en soluciones reales.

---

## Objetivos del laboratorio
- Entender la diferencia entre CPU y GPU.
- Reservar memoria en el device con `cudaMalloc`.
- Transferir datos entre host y device con `cudaMemcpy`.
- Ejecutar kernels en CUDA.
- Analizar el efecto de cambiar `grid` y `block`.
- Comprender qué pasa si falta memoria o si no se devuelve el resultado al host.
- Medir y comparar tiempos de ejecución entre CPU y GPU.

---

## Estructura del repositorio
```text
Hoja-de-Trabajo-3_Daniel-Barillas_22193_CPYD_Sec-40/
├── README.md
├── LICENSE
├── .gitignore
├── source/
│   ├── parte1_suma_cuda.cu
│   ├── parte2_umbral_clasificacion.cu
│   ├── vacio.cu
│   ├── parte3_cpu_vs_gpu.cpp
│   └── umbralizar.cu
├── scripts/
│   └── compilar.sh
├── report/
│   └── informe_ht3.tex
├── results/
│   ├── parte1_suma_cuda.txt
│   ├── parte2_umbral_clasificacion.txt
│   └── parte3_cpu_vs_gpu.txt
├── build/
└── .vscode/
```

---

## Requisitos del entorno
- WSL 2 con Ubuntu
- GPU NVIDIA compatible con CUDA
- Driver NVIDIA actualizado
- CUDA Toolkit habilitado en WSL
- Compiladores: `nvcc` y `g++`

---

## Parte 1: Suma de dos arreglos en CUDA
### Archivo
- `source/parte1_suma_cuda.cu`

### ¿Qué hace este programa?
Este ejercicio realiza la suma elemento a elemento de dos arreglos:

- Declara los arreglos `A` y `B` en host
- Inicializa ambos con valores fijos
- Reserva memoria en el device
- Copia A y B al GPU
- Ejecuta un kernel que calcula `C[i] = A[i] + B[i]`
- Copia el resultado al host y lo imprime en pantalla

### Código central
```cpp
__global__ void sumarArreglos(const int *A, const int *B, int *C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}
```

### Respuestas a las preguntas
#### 1. ¿Qué partes del código ejecutan CPU y GPU?
- CPU:
  - inicialización de arreglos
  - `cudaMalloc`
  - `cudaMemcpy`
  - impresión final
  - liberación de memoria
- GPU:
  - el kernel `sumarArreglos`
  - el cálculo real de cada elemento
  - el procesamiento paralelo de cada hilo

#### 2. ¿Qué pasa si cambia grid y block?
Si cambias el tamaño del `block` o del `grid`, cambias la distribución del trabajo.

- `block` pequeño ⇒ requiere más bloques
- `block` grande ⇒ puede saturar mejor el multiprocesador
- si `grid * block < N`, algunos elementos quedan sin procesar
- por eso se usa `if (idx < N)` para evitar accesos fuera de límites

#### 3. ¿Qué pasa si no se reserva memoria antes o después del kernel?
- sin `cudaMalloc`, el programa intenta escribir en memoria inválida
- el kernel falla o genera resultados incorrectos
- sin liberar memoria, aparecen fugas de memoria y consumo innecesario del sistema

#### 4. ¿Qué pasa si no se transfiere el resultado del GPU al CPU?
El cálculo se hace en la GPU, pero el host no tiene acceso al resultado. Entonces:
- no puedes imprimirlo
- el arreglo en host queda sin actualizar
- los valores en CPU pueden quedarse en estado viejo o indefinido

---

## Parte 2: Clasificación por umbral
### Archivos
- `source/parte2_umbral_clasificacion.cu`
- `source/vacio.cu`

### ¿Qué hace este programa?
Genera un arreglo de 32 elementos con valores `0, 5, 10, ..., 155` y luego clasifica cada valor según un umbral definido.

La regla lógica es:
```cpp
salida[idx] = (entrada[idx] >= umbral) ? 1 : 0;
```

Esto significa:
- si el valor es mayor o igual al umbral, guarda `1`
- si es menor, guarda `0`

### Resultado esperado
Con umbral `50`, se espera:
- `0` para números menores a 50
- `1` para números mayores o iguales a 50

Ejemplo:
```text
Umbral = 50
in[ 0] =   0  -> out[ 0] = 0
in[ 9] =  45  -> out[ 9] = 0
in[10] =  50  -> out[10] = 1
in[31] = 155  -> out[31] = 1
```

---

## Parte 3: Versión CPU y medición del tiempo
### Archivo
- `source/parte3_cpu_vs_gpu.cpp`

### ¿Qué hace?
Se implementa la versión CPU de la suma de arreglos y se mide el tiempo de ejecución con `std::chrono`.

### Código de medición
```cpp
auto inicioCPU = std::chrono::high_resolution_clock::now();
std::vector<int> C = sumaCPU(A, B);
auto finCPU = std::chrono::high_resolution_clock::now();

auto duracionCPU = std::chrono::duration_cast<std::chrono::microseconds>(finCPU - inicioCPU).count();
```

### Conclusión práctica
La GPU normalmente es más rápida para arreglos grandes, porque:
- cada hilo trabaja sobre un elemento distinto
- se procesa en paralelo
- la CPU recorre el conjunto secuencialmente

---

## Compilación y ejecución
### Compilar todos los programas
```bash
bash scripts/compilar.sh
```

### Compilar un programa CUDA específico
```bash
nvcc -std=c++17 -O2 source/parte1_suma_cuda.cu -o build/parte1_suma_cuda
./build/parte1_suma_cuda
```

### Compilar la parte CPU
```bash
g++ -std=c++17 -O2 source/parte3_cpu_vs_gpu.cpp -o build/parte3_cpu_vs_gpu
./build/parte3_cpu_vs_gpu
```

---

## Observaciones importantes
- WSL debe tener CUDA instalado y funcionando
- el driver de NVIDIA debe ser compatible con la versión del toolkit
- una operación GPU necesita transferencia explícita de memoria
- la configuración de `grid` y `block` influye directamente en la corrección y el rendimiento

---

## Conclusión
Este laboratorio permitió observar de forma práctica cómo la arquitectura GPU transforma la ejecución de tareas intensivas en cómputo. La CPU sigue siendo importante para lógica general y control, pero la GPU resulta ideal cuando hay muchos cálculos independientes que pueden ejecutarse simultáneamente.

La gestión correcta de memoria, la sincronización entre host y device y la elección adecuada de bloques y grids son esenciales para obtener resultados funcionales y eficientes. El trabajo demuestra no solo la capacidad de CUDA para paralelizar cálculos, sino también la importancia de comprender el flujo completo del programa para garantizar resultados correctos.
