# Enunciado oficial - Hoja de Trabajo 3

> Transcripción de apoyo. La fuente autoritativa es
> [`material_original/Hoja_de_Trabajo_3.pdf`](material_original/Hoja_de_Trabajo_3.pdf).
> Este documento no modifica ni sustituye las instrucciones del docente.

## Datos del curso

- Universidad del Valle de Guatemala
- Curso: Computación Paralela y Distribuida
- Docente: Marlon Fuentes
- Semestre: 2, 2025
- Actividad: Hoja de Trabajo #3

## Parte 1 - Suma de arreglos en CUDA

1. Crear un nuevo archivo `.cu`.
2. Escribir un programa basado en los ejemplos de clase que:
   - declare dos arreglos de tamaño `N`;
   - inicialice los arreglos `A` y `B` en el host;
   - reserve memoria en el device;
   - copie `A` y `B` hacia la GPU;
   - ejecute un kernel que sume los elementos;
   - copie el resultado al host y lo imprima.

### Preguntas requeridas

1. ¿Qué partes se ejecutan en GPU y cuáles en CPU?
2. ¿Qué ocurre al cambiar los valores de grid y block?
3. ¿Qué ocurre si no se reserva memoria antes del kernel o no se libera después?
4. ¿Qué ocurre si no se transfiere el resultado de GPU a CPU?

## Parte 2 - Clasificación por umbral

1. Descargar `vacio.cu` de la actividad.
2. Completar sus espacios en blanco para clasificar números con respecto a un
   umbral definido.

La copia original, sin completar, se conserva en
[`material_original/vacio_original.cu`](material_original/vacio_original.cu).
La solución entregable está en [`../source/vacio.cu`](../source/vacio.cu).

## Parte 3 - Versión CPU y tiempos

Realizar una versión exclusivamente CPU de cualquiera de los dos programas
anteriores. Medir el tiempo de las versiones CPU y GPU, y documentar el
resultado.

## Ponderación

| Sección | Valor |
|---|---:|
| Parte 1 | 40 % |
| Parte 2 | 35 % |
| Parte 3 | 25 % |

## Entrega indicada

El PDF original solicita entregar en Canvas:

1. Informe del proyecto en formato PDF.
2. Códigos fuente funcionales; no entregar ejecutables.

> Nota documental: el enunciado menciona “miércoles 21 de octubre” y a la vez
> “Semestre 2, 2025”. Esta transcripción conserva la información del original
> sin intentar corregir esa discrepancia de calendario.
