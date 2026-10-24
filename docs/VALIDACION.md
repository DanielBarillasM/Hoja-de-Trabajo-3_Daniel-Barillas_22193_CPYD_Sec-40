# Guía de validación y reproducibilidad

Este documento explica cómo comprobar la entrega sin depender únicamente de
las capturas incluidas en el informe.

## 1. Requisitos

- Linux o WSL 2.
- Compilador `g++` con soporte C++17.
- NVIDIA CUDA Toolkit con `nvcc` disponible en `PATH`.
- GPU NVIDIA visible mediante `nvidia-smi`.

Comprobación rápida:

```bash
g++ --version
nvcc --version
nvidia-smi
```

## 2. Ejecutar la suite completa

Desde la raíz del repositorio:

```bash
bash scripts/compilar.sh
```

El script crea `build/` y `results/`. Los ejecutables se mantienen dentro de
`build/` y están excluidos de Git; los archivos de texto de `results/` sí se
pueden conservar como evidencia reproducible.

## 3. Criterios de aceptación

### Parte 1

Para cada índice `i`:

```text
A[i] = i + 1
B[i] = 2*i + 3
C[i] = A[i] + B[i] = 3*i + 4
```

Por tanto, con `N = 16`, la salida debe comenzar en `4, 7, 10, 13` y terminar
en `49`.

### Parte 2

Con `entrada[i] = 5*i` y umbral 50:

- los índices `0..9` deben producir `0`;
- los índices `10..31` deben producir `1`.

### Parte 3

El comparador usa los mismos arreglos y el mismo tamaño para CPU y GPU.
Debe informar:

- tiempo CPU;
- tiempo GPU del kernel;
- tiempo GPU extremo a extremo (`H2D + kernel + D2H`);
- aceleración en ambos escenarios;
- cero discrepancias entre CPU, GPU y la suma esperada.

### Resultado obtenido

La ejecución final se realizó con CUDA 12.4 sobre una NVIDIA GeForce RTX 4050
Laptop GPU:

| Métrica | Resultado |
|---|---:|
| Elementos | 8,388,608 |
| Tiempo CPU | 5.106 ms |
| GPU, solo kernel | 0.662 ms |
| GPU, extremo a extremo | 11.359 ms |
| Aceleración del kernel | 7.72x |
| Aceleración extremo a extremo | 0.45x |
| Discrepancias | 0 |

La validación funcional fue correcta. La GPU aceleró el kernel, pero no el
ciclo completo debido al costo de transferir 96 MiB entre host y device.

## 4. Alcance de los tiempos

La reserva con `cudaMalloc` se excluye del tiempo extremo a extremo porque es
una preparación reutilizable. Sí se incluyen las transferencias de entrada y
salida. La métrica “solo kernel” usa eventos CUDA, que miden el trabajo en el
dispositivo sin confundirlo con la naturaleza asíncrona del lanzamiento.

Los tiempos dependen del procesador, GPU, controlador, temperatura y carga del
sistema. Por ello, no deben copiarse resultados de otra máquina.

## 5. Generar el informe

```bash
latexmk -pdf -interaction=nonstopmode \
  -outdir=output/pdf report/informe_ht3.tex
```

El resultado esperado es `output/pdf/informe_ht3.pdf`. Antes de entregar,
debe abrirse y comprobarse visualmente que no existan textos cortados,
capturas ilegibles ni referencias pendientes. La versión incluida fue
renderizada y revisada después de incorporar las mediciones finales.
