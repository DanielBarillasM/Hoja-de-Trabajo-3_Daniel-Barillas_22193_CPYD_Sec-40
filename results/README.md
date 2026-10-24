# Resultados reproducibles

Esta carpeta conserva las salidas de texto producidas por
`scripts/compilar.sh`. Los ejecutables se generan en `build/` y no se incluyen
en Git.

Para generar todos los resultados:

```bash
bash scripts/verificar_entorno.sh
bash scripts/compilar.sh
```

El archivo prioritario para completar la Parte 3 es:

```text
results/parte3_comparacion_cpu_gpu.txt
```

Debe contener tiempos CPU, GPU solo kernel, GPU extremo a extremo,
aceleraciones y una validación con cero discrepancias.
