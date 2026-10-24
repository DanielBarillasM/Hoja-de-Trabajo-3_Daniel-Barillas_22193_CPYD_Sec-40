# Resultados reproducibles

Esta carpeta conserva las salidas de texto producidas por
`scripts/compilar.sh`. Los ejecutables se generan en `build/` y no se incluyen
en Git.

Para generar todos los resultados:

```bash
bash scripts/verificar_entorno.sh
bash scripts/compilar.sh
```

Los resultados incluidos son:

- `parte1_suma_cuda.txt`: suma CUDA completa;
- `parte2_umbral_clasificacion.txt`: clasificación documentada;
- `vacio.txt`: plantilla solicitada después de completarla;
- `parte3_cpu_vs_gpu.txt`: versión CPU;
- `parte3_comparacion_cpu_gpu.txt`: tiempos CPU/GPU y validación.

La comparación final se ejecutó sobre 8,388,608 elementos en una RTX 4050
Laptop GPU. Registró 5.106 ms en CPU, 0.662 ms para el kernel GPU y 11.359 ms
para el ciclo GPU completo, con cero discrepancias.
