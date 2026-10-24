<h1 align="center">Hoja de Trabajo 3 · CUDA y paralelismo de datos</h1>

<p align="center">
  <img src="https://img.shields.io/badge/CUDA-C%2B%2B-0066FF?style=for-the-badge&logo=nvidia&logoColor=white" alt="CUDA C++" />
  <img src="https://img.shields.io/badge/Plataforma-WSL%202-1E1E1E?style=for-the-badge&logo=linux&logoColor=00FFFF" alt="WSL 2" />
  <img src="https://img.shields.io/badge/Estado-Completado-16835B?style=for-the-badge" alt="Completado" />
</p>

<p align="center">
  Implementación, análisis y documentación de suma vectorial, clasificación
  por umbral y comparación de rendimiento entre CPU y GPU.
</p>

---

## Información académica

| Campo | Información |
|---|---|
| Universidad | Universidad del Valle de Guatemala |
| Curso | Computación Paralela y Distribuida |
| Docente | Marlon Fuentes |
| Estudiante | Pablo Daniel Barillas Moreno |
| Carné | 22193 |
| Sección | 40 |
| Semestre | 2, 2026 |
| Actividad | Hoja de Trabajo #3 |

El [enunciado original](docs/material_original/Hoja_de_Trabajo_3.pdf) y la
[plantilla original `vacio.cu`](docs/material_original/vacio_original.cu) se
conservan en el repositorio como material de referencia.

## Contenido

- [Cumplimiento del enunciado](#cumplimiento-del-enunciado)
- [Estructura](#estructura)
- [Parte 1: suma CUDA](#parte-1-suma-de-arreglos-en-cuda)
- [Parte 2: clasificación](#parte-2-clasificación-por-umbral)
- [Parte 3: CPU frente a GPU](#parte-3-comparación-cpu-frente-a-gpu)
- [Ejecución](#compilación-y-ejecución)
- [Metodología](#metodología-de-medición)
- [Solución de problemas](#solución-de-problemas)

## Cumplimiento del enunciado

| Requisito | Implementación | Evidencia |
|---|---|---|
| Arreglos A y B en host | `parte1_suma_cuda.cu` | Captura y salida completa |
| Memoria en device | `cudaMalloc` para A, B y C | Errores CUDA comprobados |
| Copia host → device | `cudaMemcpyHostToDevice` | Código y mensajes de ejecución |
| Kernel de suma | `sumarArreglos` | `C[i] = A[i] + B[i]` |
| Copia device → host | `cudaMemcpyDeviceToHost` | Valores impresos |
| Respuestas conceptuales | README e informe | Análisis de Parte 1 |
| Completar `vacio.cu` | `source/vacio.cu` | Umbral 50 validado |
| Versión solo CPU | `parte3_cpu_vs_gpu.cpp` | Suma secuencial y tiempo CPU |
| Medir CPU y GPU | `parte3_comparacion_cpu_gpu.cu` | Dos métricas GPU y validación |
| Informe PDF | `output/pdf/informe_ht3.pdf` | Documento con capturas |
| No incluir ejecutables | `build/` está ignorado | Solo fuentes y documentación |

## Estructura

```text
.
├── README.md
├── LICENSE
├── docs/
│   └── material_original/
│       ├── Hoja_de_Trabajo_3.pdf
│       └── vacio_original.cu
├── source/
│   ├── parte1_suma_cuda.cu
│   ├── parte2_umbral_clasificacion.cu
│   ├── vacio.cu
│   ├── umbralizar.cu
│   ├── parte3_cpu_vs_gpu.cpp
│   └── parte3_comparacion_cpu_gpu.cu
├── scripts/
│   ├── compilar.sh
│   └── verificar_entorno.sh
├── report/
│   ├── informe_ht3.tex
│   └── imagenes/
│       ├── Archivo_vacío_Ejecución.png
│       ├── comparación_parte-3.png
│       ├── parte-1.png
│       ├── parte-2.png
│       ├── parte-3.png
│       └── verificar_entorno.png
├── output/
│   └── pdf/
│       └── informe_ht3.pdf
└── results/
    ├── parte1_suma_cuda.txt
    ├── parte2_umbral_clasificacion.txt
    ├── parte3_comparacion_cpu_gpu.txt
    ├── parte3_cpu_vs_gpu.txt
    └── vacio.txt
```

`build/` se genera localmente y no forma parte de la entrega.

## Parte 1: suma de arreglos en CUDA

Archivo: [`source/parte1_suma_cuda.cu`](source/parte1_suma_cuda.cu).

El host crea dos arreglos de 16 elementos:

```text
A[i] = i + 1
B[i] = 2*i + 3
```

Cada hilo calcula un elemento independiente:

```cpp
__global__ void sumarArreglos(const int *A, const int *B, int *C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}
```

El resultado esperado es `C[i] = 3*i + 4`: `4, 7, 10, 13, ..., 49`.

### Respuestas solicitadas

#### ¿Qué se ejecuta en CPU y qué se ejecuta en GPU?

La CPU inicializa los arreglos, solicita memoria del dispositivo, ordena las
copias, configura y lanza el kernel, recupera la salida y la imprime. La GPU
ejecuta `sumarArreglos`: cada hilo calcula su índice global y realiza una suma.
Las llamadas `cudaMalloc` y `cudaMemcpy` son invocadas por el host.

#### ¿Qué ocurre al cambiar grid y block?

`gridSize * blockSize` determina cuántos hilos se crean. Si es menor que `N`,
quedan elementos sin procesar. Si es mayor, `idx < N` impide accesos fuera del
arreglo. Un bloque más grande no garantiza más velocidad: también influyen los
límites de hilos, registros, memoria compartida y ocupación.

La división de techo cubre todos los elementos:

```cpp
int gridSize = (N + blockSize - 1) / blockSize;
```

#### ¿Qué ocurre si no se administra correctamente la memoria?

Sin `cudaMalloc`, los punteros del device no representan almacenamiento válido
y las copias o el kernel fallan. Liberar memoria antes de terminar provoca
accesos inválidos; no liberarla después produce fugas de memoria.

#### ¿Qué ocurre si no se copia el resultado a CPU?

`d_C` y `h_C` pertenecen a espacios diferentes. Sin una copia
`DeviceToHost`, el resultado permanece en la GPU y el host no puede imprimirlo
ni reutilizarlo como un arreglo actualizado.

## Parte 2: clasificación por umbral

La solución requerida está en [`source/vacio.cu`](source/vacio.cu). La
plantilla original se conserva en
[`docs/material_original/vacio_original.cu`](docs/material_original/vacio_original.cu).

Para `N = 32`, `entrada[i] = 5*i` y `UMBRAL = 50`:

```cpp
salida[idx] = (entrada[idx] >= umbral) ? 1 : 0;
```

Los valores de 0 a 45 producen `0`; desde 50 hasta 155 producen `1`.
`parte2_umbral_clasificacion.cu` ofrece la misma solución con mensajes
explicativos y `umbralizar.cu` conserva nombres próximos a la plantilla.

## Parte 3: comparación CPU frente a GPU

La Parte 3 se divide en dos archivos para evidenciar ambos requisitos:

- [`parte3_cpu_vs_gpu.cpp`](source/parte3_cpu_vs_gpu.cpp): versión
  exclusivamente CPU.
- [`parte3_comparacion_cpu_gpu.cu`](source/parte3_comparacion_cpu_gpu.cu):
  comparación sobre los mismos 8,388,608 elementos.

El comparador informa:

1. tiempo CPU con `std::chrono::steady_clock`;
2. tiempo exclusivo del kernel mediante eventos CUDA;
3. tiempo GPU extremo a extremo (`H2D + kernel + D2H`);
4. aceleración respecto de CPU;
5. validación elemento por elemento.

### Resultado medido

La prueba se ejecutó en una NVIDIA GeForce RTX 4050 Laptop GPU con CUDA 12.4:

| Métrica | Resultado |
|---|---:|
| Elementos | 8,388,608 |
| Tiempo CPU | 5.106 ms |
| GPU, solo kernel | 0.662 ms |
| GPU, incluyendo H2D + kernel + D2H | 11.359 ms |
| Aceleración del kernel | 7.72x |
| Aceleración extremo a extremo | 0.45x |
| Discrepancias | 0 |

El kernel GPU fue 7.72 veces más rápido que la CPU. Sin embargo, al incluir
las transferencias, la ejecución GPU completa tardó aproximadamente 2.22 veces
lo que tardó la CPU. Este resultado muestra que paralelizar el cálculo no
garantiza acelerar el programa completo cuando la comunicación domina el costo.

## Compilación y ejecución

### 1. Requisitos del sistema

Para compilar y ejecutar todos los programas se necesita:

- WSL 2 con Ubuntu o una distribución Linux equivalente;
- GPU NVIDIA compatible con CUDA;
- controlador NVIDIA visible desde WSL;
- NVIDIA CUDA Toolkit con `nvcc` disponible en `PATH`;
- `g++` con soporte para C++17;
- `latexmk` y una distribución LaTeX, únicamente para regenerar el informe.

Comprueba las herramientas principales:

```bash
g++ --version
nvcc --version
nvidia-smi
```

El entorno usado para los resultados incluidos fue CUDA 12.4 con una NVIDIA
GeForce RTX 4050 Laptop GPU.

### 2. Obtener y abrir el repositorio

```bash
git clone https://github.com/DanielBarillasM/Hoja-de-Trabajo-3_Daniel-Barillas_22193_CPYD_Sec-40.git
cd Hoja-de-Trabajo-3_Daniel-Barillas_22193_CPYD_Sec-40
```

Si ya existe una copia local:

```bash
git pull origin main
```

### 3. Verificar el entorno

```bash
bash scripts/verificar_entorno.sh
```

El script debe mostrar `[OK]` para `g++`, `nvcc` y `nvidia-smi`, además del
modelo de GPU detectado. Si alguna comprobación falla, no ejecutes la suite
hasta corregirla.

### 4. Compilar y ejecutar la entrega completa

```bash
bash scripts/compilar.sh
```

Este comando:

1. crea `build/` y `results/`;
2. compila la suma CUDA de la Parte 1;
3. compila las soluciones de clasificación de la Parte 2;
4. compila la versión secuencial CPU;
5. compila y ejecuta la comparación CPU/GPU;
6. guarda cada salida en un archivo `.txt` dentro de `results/`.

Los ejecutables quedan en `build/`, que está excluido mediante `.gitignore` y
no debe incluirse en la entrega.

### 5. Compilación manual por programa

```bash
mkdir -p build results

# Parte 1: suma CUDA
nvcc -std=c++17 -O2 source/parte1_suma_cuda.cu \
  -o build/parte1_suma_cuda
./build/parte1_suma_cuda | tee results/parte1_suma_cuda.txt

# Parte 2: clasificación por umbral
nvcc -std=c++17 -O2 source/parte2_umbral_clasificacion.cu \
  -o build/parte2_umbral_clasificacion
./build/parte2_umbral_clasificacion \
  | tee results/parte2_umbral_clasificacion.txt

# Parte 2: plantilla vacio.cu completada
nvcc -std=c++17 -O2 source/vacio.cu -o build/vacio
./build/vacio | tee results/vacio.txt

# Parte 3: versión exclusivamente CPU
g++ -std=c++17 -O2 source/parte3_cpu_vs_gpu.cpp \
  -o build/parte3_cpu_vs_gpu
./build/parte3_cpu_vs_gpu | tee results/parte3_cpu_vs_gpu.txt

# Parte 3: comparación CPU/GPU
nvcc -std=c++17 -O2 source/parte3_comparacion_cpu_gpu.cu \
  -o build/parte3_comparacion_cpu_gpu
./build/parte3_comparacion_cpu_gpu \
  | tee results/parte3_comparacion_cpu_gpu.txt
```

`source/umbralizar.cu` es una variante equivalente de la Parte 2. Puede
compilarse por separado con:

```bash
nvcc -std=c++17 -O2 source/umbralizar.cu -o build/umbralizar
./build/umbralizar
```

### 6. Regenerar el informe PDF

Desde la raíz del repositorio:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error \
  -outdir=output/pdf report/informe_ht3.tex
```

El archivo generado queda en
[`output/pdf/informe_ht3.pdf`](output/pdf/informe_ht3.pdf).

## Metodología de medición

CPU y GPU procesan la misma operación, valores y tamaño. `cudaMalloc` queda
fuera del tiempo extremo a extremo por ser preparación reutilizable; las
transferencias sí se incluyen. También se informa el kernel por separado para
distinguir cómputo de comunicación.

Se realiza un calentamiento antes de medir el kernel, evitando que la creación
del contexto CUDA distorsione el resultado. CPU y GPU se comparan contra
`A[i] + B[i]`; el programa termina con error si encuentra discrepancias.

## Evidencia visual

| Parte | Captura | Verificación |
|---|---|---|
| 1 | [`parte-1.png`](report/imagenes/parte-1.png) | Secuencia `4..49` correcta |
| 2 | [`parte-2.png`](report/imagenes/parte-2.png) | Cambio de 0 a 1 en 50 |
| Plantilla | [`Archivo_vacío_Ejecución.png`](report/imagenes/Archivo_vacío_Ejecución.png) | `vacio.cu` completado |
| Entorno | [`verificar_entorno.png`](report/imagenes/verificar_entorno.png) | CUDA 12.4 y RTX 4050 detectados |
| CPU/GPU | [`parte-3.png`](report/imagenes/parte-3.png) | Suite completa y validación correcta |
| Comparación | [`comparación_parte-3.png`](report/imagenes/comparación_parte-3.png) | Tiempos, aceleraciones y 0 discrepancias |

## Solución de problemas

### `nvcc: command not found`

CUDA Toolkit no está instalado o su carpeta `bin` no está en `PATH`. Ejecuta
`nvcc --version` y corrige la instalación antes de compilar.

### `nvidia-smi` no detecta la GPU en WSL

Verifica que WSL 2 esté activo y que Windows tenga un controlador NVIDIA con
soporte para CUDA en WSL. Reinicia WSL después de actualizar el controlador.

### `$'\r': command not found`

El script tiene finales de línea Windows. El repositorio incluye
`.gitattributes` para forzar LF; después de actualizar, restaura los scripts:

```bash
git pull origin main
git restore --source=HEAD --worktree scripts/compilar.sh scripts/verificar_entorno.sh
```

### Limpiar y recompilar

```bash
rm -rf build
bash scripts/compilar.sh
```

## Conclusión

El repositorio demuestra el flujo completo de CUDA: preparación en host,
memoria explícita, lanzamiento del kernel, sincronización, retorno de resultados
y validación. La medición confirma una ventaja de 7.72x dentro del kernel, pero
también muestra que las transferencias reducen la aceleración extremo a extremo
a 0.45x. Por ello, el beneficio de la GPU depende tanto del paralelismo como de
la cantidad de comunicación necesaria.
