# Resumen 3 / Professional CUDA C Programming
## 'Checking the data layout of shared memory'

Esta seccion detalla como utilizar la memoria compartida de manera efectiva, utilizando comparaciones de estructuras de datos y conceptos basicos que uno debe tomar en cuenta al desarrolar programas que utilizen memoria compartida.
Especificamente:
- Arreglos cuadrados y arreglos rectangulares
- Acceso de fila mayor y columna mayor
- Declaraciones dinamicas y estaticas
- Alcance de archivo y kernel
- Con y sin desplazamiento de memoria

## Memoria Compartida en configuración cuadrada:
El primer apartado de la seccion habla sobre el uso de memoria compartida en una configuracion de matriz cuadrada, la cual debido a su simplicidad hace facil calcular desplazamientos de memoria 1D (unidimensionales) dentro de indices de hilos 2D.
La variable de memoria compartida de dos dimensiones de puede declarar en un codigo de la siguiente manera: `__shared__ int tile [N] [N] ;`, lo importante de esta declaracion de memoria, siendo una configuracion cuadrada, es que se puede acceder a ella desde un bloque de hilos cuadrado, con los hilos vecinos accediendo a los valores vecinos en las dimensiones `x` o `y` de la siguiente manera:

* `tile [threadIdx.y] [threadIdx.x]` / Accede a los valores vecinos de una fila (x)
* `tile [threadIdx.x] [threadIdx.y]` / Accede a los valores vecinos de una columna (y)

![SquareSharedMemory]("../../Resources/SquareSharedMemory.png")

Ya que acceder a valores del mismo banco en diferentes filas causa un conflicto, disminuyendo la eficiencia del programa dado que tiene que dividir y repetir la solicitud de los hilos correspondientes en cuantos ciclos sean necesarios, podemos determinar que el metodo de acceso `tile [threadIdx.y] [threadIdx.x]` es mas eficiente dado que accede a valores en diferentes bancos en la misma fila.

## Acceso en orden principal de fila y principal de columna:
Estos son metodos para almacenar arreglos multidimensionales en almacenamiento lineal como la RAM, cuya diferencia entre ambos son los elementos que son vecinos en memoria.

![RowColumnMajorOrder]("../../../Resources/OrdenPrincipalColumnaFila.png")



