#!/usr/bin/env bash
set -u

estado=0

comprobar() {
    local comando="$1"
    local descripcion="$2"

    if command -v "$comando" >/dev/null 2>&1; then
        printf "[OK] %-12s %s\n" "$comando" "$descripcion"
    else
        printf "[FALTA] %-9s %s\n" "$comando" "$descripcion" >&2
        estado=1
    fi
}

printf "Verificacion del entorno de Hoja de Trabajo 3\n"
printf "============================================================\n"

comprobar g++ "Compilador para la version CPU"
comprobar nvcc "Compilador NVIDIA CUDA"
comprobar nvidia-smi "Acceso al controlador y a la GPU NVIDIA"

if command -v nvcc >/dev/null 2>&1; then
    printf "\nVersion de CUDA:\n"
    nvcc --version | tail -n 1
fi

if command -v nvidia-smi >/dev/null 2>&1; then
    printf "\nGPU detectada:\n"
    nvidia-smi --query-gpu=name,driver_version,memory.total \
        --format=csv,noheader
fi

if [[ "$estado" -eq 0 ]]; then
    printf "\n[OK] El entorno esta listo. Ejecuta: bash scripts/compilar.sh\n"
else
    printf "\n[ERROR] Instala o habilita los componentes marcados antes de continuar.\n" >&2
fi

exit "$estado"
