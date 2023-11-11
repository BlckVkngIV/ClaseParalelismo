# Resultados del Nvidia Profiler en codigos CUDA
# Professional CUDA C / Chapter 04
## Índice:
### [WriteSegment.cu](#writesegmentcu-1)
### [SumArrayZeroCpy.cu](#sumarrayzerocpycu-1)
### [SimpleMathAoS.cu](#simplemathaoscu-1)
### [simpleMathSoA.cu](#simplemathsoacu-1)
### [memTranfer.cu](#memtranfercu-1)
### [pinMemTransfer.cu](#pinmemtransfercu-1)
### [sumMatrixGPUManaged.cu](#summatrixgpumanagedcu-1)
### [summatrixGPUManual.cu](#summatrixgpumanualcu-1)
### [readSegment.cu](#readsegmentcu-1)
### [readSegmentUnroll.cu](#readsegmentunrollcu-1)


# writeSegment.cu
Este ejemplo muestra el impacto que tienen las escrituras desalineadas en rendimiento forzando escrituras desalineadas en una variable flotante.

```           Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   65.98%  2.1129ms         3  704.29us  518.98us  921.45us  [CUDA memcpy DtoH]
                   29.36%  940.23us         2  470.12us  465.19us  475.04us  [CUDA memcpy HtoD]
                    1.55%  49.504us         1  49.504us  49.504us  49.504us  writeOffset(float*, float*, float*, int, int)
                    1.49%  47.712us         1  47.712us  47.712us  47.712us  warmup(float*, float*, float*, int, int)
                    0.91%  29.120us         1  29.120us  29.120us  29.120us  writeOffsetUnroll2(float*, float*, float*, int, int)
                    0.72%  23.072us         1  23.072us  23.072us  23.072us  writeOffsetUnroll4(float*, float*, float*, int, int)
      API calls:   92.61%  579.23ms         3  193.08ms  301.40us  578.59ms  cudaMalloc
                    6.01%  37.576ms         1  37.576ms  37.576ms  37.576ms  cudaDeviceReset
                    0.83%  5.1802ms         5  1.0360ms  537.40us  2.0100ms  cudaMemcpy
                    0.34%  2.1550ms         1  2.1550ms  2.1550ms  2.1550ms  cuDeviceGetPCIBusId
                    0.11%  687.50us         3  229.17us  186.80us  276.10us  cudaFree
                    0.06%  399.00us         4  99.750us  72.600us  145.30us  cudaDeviceSynchronize
                    0.04%  225.60us         4  56.400us  20.100us  89.100us  cudaLaunchKernel
                    0.00%  14.400us       101     142ns     100ns  1.0000us  cuDeviceGetAttribute
                    0.00%  4.9000us         1  4.9000us  4.9000us  4.9000us  cudaGetDeviceProperties
                    0.00%  4.1000us         1  4.1000us  4.1000us  4.1000us  cudaSetDevice
                    0.00%  3.3000us         4     825ns     400ns  1.1000us  cudaGetLastError
                    0.00%  1.2000us         3     400ns     200ns     800ns  cuDeviceGetCount
                    0.00%  1.1000us         2     550ns     200ns     900ns  cuDeviceGet
                    0.00%     700ns         1     700ns     700ns     700ns  cuDeviceGetName
                    0.00%     300ns         1     300ns     300ns     300ns  cuDeviceTotalMem
                    0.00%     100ns         1     100ns     100ns     100ns  cuDeviceGetUuid
```

### Descripción de resultados.
De acuerdo al NVProfiler, se muestra que la GPU utilizó la mayor parte del tiempo copiando memoria del dispositivo al host, usando el 65.98% del tiempo total en 3
llamadas a esta función, seguidas por copias de memoria del host al dispositivo, las cuales utilizaron mucho menos tiempo, 
solamente un 29.36% del tiempo total en 2 llamadas. Las siguientes 4 llamadas en 'GPU Activities' corresponden a las funciones que realizan las escrituras desalineadas forzadas y corresponden a un 4.67% del tiempo total. 
La gran diferencia se debe a que estas operaciones se ejecutan directamente en los multiples nucleos de la grafica,
y las otras 2 funciones requieren de comunicación con el host, por lo cual se ven afectadas por los limites de velocidad de los dispositivos.
En cambio en la seccion de 'API calls' podemos ver como 3 llamadas a 'cudaMalloc' abarcan el 92.61% del tiempo total, esto debido a que la primer llamada a una 
función CUDA inicializa todo el subsistema de CUDA, por lo cual tomó el mayor tiempo con 578.59ms, ya inicializado el subsistema, 
las siguientes llamadas a métodos CUDA tomarán mucho menos tiempo, esto visto en las restantes 2 llamadas al metodo, 
cuyo menor tiempo fue de 301.40 μs (microsegundos) al igual que el resto de llamadas a CUDA (134) que abarcaron un 7.28% del tiempo total.

# sumArrayZerocpy.cu
Este ejemplo demuestra el uso de memoria copia cero para evitar el uso de una operación 'memcpy' entre el host y el dispositivo. Haciendo una referencia directa al host, se pueden traspasar los resultados a traves del PCIe bus.

```
	Type  	Time(%)      Time     Calls       Avg       Min       Max  	Name
 GPU activities:   33.33%  3.5200us         1  3.5200us  3.5200us  3.5200us  sumArraysZeroCopy(float*, float*, float*, int)
                   22.73%  2.4000us         2  1.2000us  1.1840us  1.2160us  [CUDA memcpy DtoH]
                   22.12%  2.3360us         1  2.3360us  2.3360us  2.3360us  sumArrays(float*, float*, float*, int)
                   21.82%  2.3040us         2  1.1520us     864ns  1.4400us  [CUDA memcpy HtoD]
      API calls:   94.24%  583.14ms         3  194.38ms  1.8000us  583.14ms  cudaMalloc
                    5.09%  31.475ms         1  31.475ms  31.475ms  31.475ms  cudaDeviceReset
                    0.35%  2.1756ms         1  2.1756ms  2.1756ms  2.1756ms  cuDeviceGetPCIBusId
                    0.16%  988.60us         2  494.30us  3.8000us  984.80us  cudaHostAlloc
                    0.06%  368.90us         2  184.45us  4.5000us  364.40us  cudaFreeHost
                    0.06%  358.00us         4  89.500us  33.100us  129.40us  cudaMemcpy
                    0.04%  218.20us         3  72.733us  2.5000us  208.10us  cudaFree
                    0.01%  60.300us         2  30.150us  28.600us  31.700us  cudaLaunchKernel
                    0.00%  14.900us       101     147ns     100ns  1.0000us  cuDeviceGetAttribute
                    0.00%  6.2000us         1  6.2000us  6.2000us  6.2000us  cudaSetDevice
                    0.00%  2.3000us         1  2.3000us  2.3000us  2.3000us  cudaGetDeviceProperties
                    0.00%  2.1000us         2  1.0500us     600ns  1.5000us  cudaHostGetDevicePointer
                    0.00%  1.6000us         3     533ns     200ns  1.0000us  cuDeviceGetCount
                    0.00%  1.1000us         2     550ns     200ns     900ns  cuDeviceGet
                    0.00%     700ns         1     700ns     700ns     700ns  cuDeviceGetName
                    0.00%     300ns         1     300ns     300ns     300ns  cuDeviceTotalMem
                    0.00%     200ns         1     200ns     200ns     200ns  cuDeviceGetUuid
```
### Descripción de resultados.
Aqui podemos ver como la función que mas tiempo abarca (en %), es 'sumArraysZeroCopy', utilizando 3.5200 μs para toda la función, a pesar de lo que se muestra, 
esta función es más eficiente que 'sumArrays' ya que esta utiliza el 'mempcy' de host a dispositivo y viceversa, sumando las 3 funciones, 
abarcan un 66.68% del tiempo total, o 7.4 μs para todo el proceso de la función y la tranferencia de datos entre host y dispositivo.
Esto se debe a la referencia directa guardada del host, lo cual ahorra el tiempo usado en las funciones 'memcpy' a pesar de ser mas lenta que la función original.

# simpleMathAoS.cu
Este codigo es un ejemplo de usar arreglos de estructuras para guardar datos en el dispositivo, para estudiar el impacto de la estructura de datos en el desempeño.
```
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   80.19%  23.304ms         2  11.652ms  8.6804ms  14.623ms  [CUDA memcpy DtoH]
                   18.05%  5.2457ms         1  5.2457ms  5.2457ms  5.2457ms  [CUDA memcpy HtoD]
                    0.88%  256.10us         1  256.10us  256.10us  256.10us  warmup(innerStruct*, innerStruct*, int)
                    0.88%  255.85us         1  255.85us  255.85us  255.85us  testInnerStruct(innerStruct*, innerStruct*, int)
      API calls:   88.83%  566.38ms         2  283.19ms  376.90us  566.01ms  cudaMalloc
                    5.61%  35.742ms         1  35.742ms  35.742ms  35.742ms  cudaDeviceReset
                    4.90%  31.267ms         3  10.422ms  6.6946ms  15.576ms  cudaMemcpy
                    0.34%  2.1562ms         1  2.1562ms  2.1562ms  2.1562ms  cuDeviceGetPCIBusId
                    0.19%  1.2200ms         2  610.00us  446.90us  773.10us  cudaFree
                    0.11%  669.90us         2  334.95us  334.90us  335.00us  cudaDeviceSynchronize
                    0.03%  161.60us         2  80.800us  63.500us  98.100us  cudaLaunchKernel
                    0.00%  14.600us       101     144ns     100ns  1.3000us  cuDeviceGetAttribute
                    0.00%  6.1000us         1  6.1000us  6.1000us  6.1000us  cudaSetDevice
                    0.00%  5.8000us         2  2.9000us  2.6000us  3.2000us  cudaGetLastError
                    0.00%  4.7000us         1  4.7000us  4.7000us  4.7000us  cudaGetDeviceProperties
                    0.00%  1.4000us         3     466ns     200ns  1.0000us  cuDeviceGetCount
                    0.00%     800ns         2     400ns     100ns     700ns  cuDeviceGet
                    0.00%     600ns         1     600ns     600ns     600ns  cuDeviceGetName
                    0.00%     400ns         1     400ns     400ns     400ns  cuDeviceTotalMem
                    0.00%     200ns         1     200ns     200ns     200ns  cuDeviceGetUuid
```
## Descripción de resultados:
El tiempo utilizado durante el proceso fue mayoritariamente usado por las funciones 'memcpy' (DtoH y HtoD), que se encargan de copiar datos del dispositivo al host (DtoH) y viceversa (HtoD), las cuales abarcaron en conjunto un 98.24% a lo largo de 3 llamadas (2 DtoH y 1 HtoD) del uso de la tarjeta grafica, siendo seguidas en desempeño por la función 'warmup' que sirve para agregar a los componentes 'x' e 'y' de la estructura de datos y abarcó un 0.88% del tiempo, al igual que la función 'testInnerStruct', ambas solo fueron llamadas 1 vez.
En las llamadas realizadas a la API, 'cudaMalloc' la función encargada de asignar memoria en la GPU, fue la que más tiempo abarcó del proceso, esto debido a la inicializacion del subsistema CUDA (descrito en resultados anteriores), usando 88.83% del tiempo en 2 llamadas, seguida por la función 'cudaDeviceReset' encargada de restablecer y liberar memoria, usando 5.61% en 1 llamada, 'cudaMemcpy', que tomó 3 llamadas mencionadas anteriormente en la descripción de actividades de GPU, 'cuDeviceGetPCIBusID' encargada de regresar una linea de caracteres de acuerdo al Bus PCI del dispositivo usando 0.34% del tiempo en 1 llamada, 'cudaFree' encargada de liberar memoria asignada con un 0.19% de uso en 2 llamadas, 'cudaDeviceSynchronize', con un 0.11% de uso en 2 llamadas, 'cuDeviceGetAttribute' encargada de regresar información sobre el dispositivo (GPU), llamada 101 veces usando 14.600 microsegundos y 'cudaSetDevice' la cual establece el dispositivo a usar para las operaciones.

# simpleMathSoA.cu
Similar al codigo anterior, utiliza arreglos de estructuras para guardar datos, estudiando su impacto mediante leidas continuas.
```
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   73.35%  12.215ms         2  6.1076ms  3.7599ms  8.4554ms  [CUDA memcpy DtoH]
                   23.58%  3.9265ms         1  3.9265ms  3.9265ms  3.9265ms  [CUDA memcpy HtoD]
                    1.54%  256.42us         1  256.42us  256.42us  256.42us  warmup2(InnerArray*, InnerArray*, int)
                    1.54%  256.03us         1  256.03us  256.03us  256.03us  testInnerArray(InnerArray*, InnerArray*, int)
      API calls:   90.98%  584.89ms         2  292.45ms  380.00us  584.51ms  cudaMalloc
                    5.47%  35.165ms         1  35.165ms  35.165ms  35.165ms  cudaDeviceReset
                    2.89%  18.564ms         3  6.1881ms  3.9129ms  9.2690ms  cudaMemcpy
                    0.39%  2.4897ms         1  2.4897ms  2.4897ms  2.4897ms  cuDeviceGetPCIBusId
                    0.15%  981.80us         2  490.90us  359.80us  622.00us  cudaFree
                    0.11%  682.20us         2  341.10us  302.90us  379.30us  cudaDeviceSynchronize
                    0.01%  94.200us         2  47.100us  43.700us  50.500us  cudaLaunchKernel
                    0.00%  16.500us       101     163ns     100ns  1.4000us  cuDeviceGetAttribute
                    0.00%  5.9000us         1  5.9000us  5.9000us  5.9000us  cudaSetDevice
                    0.00%  5.1000us         1  5.1000us  5.1000us  5.1000us  cudaGetDeviceProperties
                    0.00%  4.7000us         2  2.3500us  2.3000us  2.4000us  cudaGetLastError
                    0.00%  1.4000us         3     466ns     200ns  1.0000us  cuDeviceGetCount
                    0.00%  1.2000us         1  1.2000us  1.2000us  1.2000us  cuDeviceGetName
                    0.00%  1.1000us         2     550ns     100ns  1.0000us  cuDeviceGet
                    0.00%     300ns         1     300ns     300ns     300ns  cuDeviceTotalMem
                    0.00%     200ns         1     200ns     200ns     200ns  cuDeviceGetUuid
```
## Descripción de Resultados:
Debido a que este codigo es muy similar al anterior, nuevamente la operacion de copia de memoria desde el dispositivo al host fué la que más tiempo tomó, con 73.35% del tiempo en 2 llamadas, seguida por la copia del host al dispositivo con 23.58% en 1 llamada y por las funciones 'warmup2' y 'testInnerArray' encargadas de realizar los procesos de arreglos del codigo, usando 1.54% en 1 llamada cada función.
Nuevamente en las llamadas a la API, 'cudaMalloc' siendo la primer y mas tardada llamada debido a la inicialización del susbsistema usando 90.98% en dos llamadas

# memTransfer.cu
Ejemplo de la copia de memoria de la API CUDA para transferir datos del dispositivo al host y vicecersa.
```
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   52.15%  2.1117ms         1  2.1117ms  2.1117ms  2.1117ms  [CUDA memcpy HtoD]
                   47.85%  1.9374ms         1  1.9374ms  1.9374ms  1.9374ms  [CUDA memcpy DtoH]
      API calls:   93.74%  577.35ms         1  577.35ms  577.35ms  577.35ms  cudaMalloc
                    5.15%  31.729ms         1  31.729ms  31.729ms  31.729ms  cudaDeviceReset
                    0.71%  4.3856ms         2  2.1928ms  2.1784ms  2.2072ms  cudaMemcpy
                    0.34%  2.0994ms         1  2.0994ms  2.0994ms  2.0994ms  cuDeviceGetPCIBusId
                    0.05%  306.30us         1  306.30us  306.30us  306.30us  cudaFree
                    0.00%  14.700us       101     145ns     100ns  1.0000us  cuDeviceGetAttribute
                    0.00%  8.0000us         1  8.0000us  8.0000us  8.0000us  cudaSetDevice
                    0.00%  2.8000us         1  2.8000us  2.8000us  2.8000us  cudaGetDeviceProperties
                    0.00%  1.4000us         3     466ns     200ns  1.0000us  cuDeviceGetCount
                    0.00%     900ns         2     450ns     100ns     800ns  cuDeviceGet
                    0.00%     900ns         1     900ns     900ns     900ns  cuDeviceGetName
                    0.00%     400ns         1     400ns     400ns     400ns  cuDeviceTotalMem
                    0.00%     200ns         1     200ns     200ns     200ns  cuDeviceGetUuid
```
## Descripción de Resultados
Este codigo solo hace uso de las funciones de copia de memoria en la GPU, usando un 52.15% (2.1117ms) para la copia de host a dispositivo y un 47.85% (1.9374ms) para las copias del dispositivo al host.
Dentro de las llamadas a API también son relativamente pocas, usando solo 1 llamada a 'cudaMalloc' para inicialización de subsistema y asignación de memoria con 93.74% del tiempo, 'cudaDeviceReset' para reestablecer el dispositivo con 1 llamada usando 5.15%, 2 llamadas a 'cudaMemcpy' para la copia de memoria mencionada anteriormente usando 0.71% del tiempo, 'cuDeviceGetPCIBusId' para encontrar el Bus PCI asignado al dispositivo (0.34%, 1 llamada), 'cudaFree' para liberación de memoria (0.05%, 1 llamada), 'cuDeviceGetAttribute' para recibir los atributos especificos del dispositivo a usar, 'cudaSetDevice' para definir el dispositivo a usar, 'cudaGetDeviceProperties' para determinar las propiedades del dispositivo, 'cuDeviceGetCount' para establecer el numero de dispositivos disponibles, 'cuDeviceGet' que regresa el identificador del dispositivo, 'cuDeviceGetName' que regresa el nombre del dispositivo ingresado, 'cuDeviceTotalMem' que regresa la capacidad de memoria total del dispositivo, 'cuDeviceGetUuid' regresa el identificador unico universal del dispositivo ingresado.

# pinMemTransfer.cu
Similar al codigo anterior, utiliza la API para la transferencia de datos entre el dispositivo y el host, con la diferencia de que se utiliza 'cudaMallocHost' para crear un arreglo dentro del host.
```
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   50.57%  1.3036ms         1  1.3036ms  1.3036ms  1.3036ms  [CUDA memcpy HtoD]
                   49.43%  1.2743ms         1  1.2743ms  1.2743ms  1.2743ms  [CUDA memcpy DtoH]
      API calls:   93.65%  564.84ms         1  564.84ms  564.84ms  564.84ms  cudaHostAlloc
                    5.15%  31.051ms         1  31.051ms  31.051ms  31.051ms  cudaDeviceReset
                    0.45%  2.7319ms         2  1.3660ms  1.3368ms  1.3951ms  cudaMemcpy
                    0.34%  2.0604ms         1  2.0604ms  2.0604ms  2.0604ms  cuDeviceGetPCIBusId
                    0.30%  1.8091ms         1  1.8091ms  1.8091ms  1.8091ms  cudaFreeHost
                    0.06%  342.90us         1  342.90us  342.90us  342.90us  cudaMalloc
                    0.04%  261.00us         1  261.00us  261.00us  261.00us  cudaFree
                    0.00%  15.400us       101     152ns     100ns     900ns  cuDeviceGetAttribute
                    0.00%  7.2000us         1  7.2000us  7.2000us  7.2000us  cudaSetDevice
                    0.00%  2.4000us         1  2.4000us  2.4000us  2.4000us  cudaGetDeviceProperties
                    0.00%  1.0000us         3     333ns     100ns     700ns  cuDeviceGetCount
                    0.00%  1.0000us         2     500ns     100ns     900ns  cuDeviceGet
                    0.00%     600ns         1     600ns     600ns     600ns  cuDeviceGetName
                    0.00%     400ns         1     400ns     400ns     400ns  cuDeviceTotalMem
                    0.00%     200ns         1     200ns     200ns     200ns  cuDeviceGetUuid
```
## Descripción de resultados:
Este codigo, al igual que el anterior solo hace uso de las funciones de copia de memoria en la GPU, usando un 50.57% (1.3036ms) para la copia de host a dispositivo y un 49.43% (1.2743mx) para las copias del dispositivo al host.
Dentro de las llamadas a API también son relativamente pocas, usando solo 1 llamada a 'cudaMallocHost' para inicialización de subsistema y asignación de memoria dentro del host con 93.65% del tiempo, 'cudaDeviceReset' para reestablecer el dispositivo con 1 llamada usando 5.15%, 2 llamadas a 'cudaMemcpy' para la copia de memoria mencionada anteriormente usando 0.45% del tiempo, 'cuDeviceGetPCIBusId' para encontrar el Bus PCI asignado al dispositivo (0.34%, 1 llamada), 'cudaFreeHost' para la liberación de memoria asignada al host (0.30%, 1 llamada). Mientras que la mayoria de la asignacion de memoria ocurre en el host, tambien de usa 'cudaMalloc' y 'cudaFree' para asignar y liberar memoria en el dispositivo GPU, usando 0.06% con 1 llamada y 0.04% con 1 llamada respectivamente.
El codigo tambien hace las llamadas normales de 'cuDeviceGetAttribute' para recibir los atributos especificos del dispositivo a usar, 'cudaSetDevice' para definir el dispositivo a usar, 'cudaGetDeviceProperties' para determinar las propiedades del dispositivo, 'cuDeviceGetCount' para establecer el numero de dispositivos disponibles, 'cuDeviceGet' que regresa el identificador del dispositivo, 'cuDeviceGetName' que regresa el nombre del dispositivo ingresado, 'cuDeviceTotalMem' que regresa la capacidad de memoria total del dispositivo, 'cuDeviceGetUuid' regresa el identificador unico universal del dispositivo ingresado.

# sumMatrixGPUManaged.cu
Este codigo usa la memoria CUDA para hacer sumas de matrices utilizando punteros, CUDA automaticamente hace la transferencia de datos de acuerdo a las necesidades de la aplicación por ello es innecesario usar alguna llamada explicita para transferencia de datos.
```
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:  100.00%  12.948ms         2  6.4741ms  288.67us  12.660ms  sumMatrixGPU(float*, float*, float*, int, int)
      API calls:   91.39%  815.38ms         4  203.85ms  27.532ms  731.17ms  cudaMallocManaged
                    3.45%  30.801ms         1  30.801ms  30.801ms  30.801ms  cudaDeviceReset
                    3.31%  29.569ms         4  7.3922ms  7.2484ms  7.4490ms  cudaFree
                    1.52%  13.583ms         1  13.583ms  13.583ms  13.583ms  cudaDeviceSynchronize
                    0.24%  2.1681ms         1  2.1681ms  2.1681ms  2.1681ms  cuDeviceGetPCIBusId
                    0.07%  644.20us         2  322.10us  11.200us  633.00us  cudaLaunchKernel
                    0.00%  14.100us       101     139ns     100ns     900ns  cuDeviceGetAttribute
                    0.00%  5.8000us         1  5.8000us  5.8000us  5.8000us  cudaSetDevice
                    0.00%  4.0000us         1  4.0000us  4.0000us  4.0000us  cudaGetDeviceProperties
                    0.00%  1.2000us         3     400ns     100ns     900ns  cuDeviceGetCount
                    0.00%  1.1000us         2     550ns     100ns  1.0000us  cuDeviceGet
                    0.00%     900ns         1     900ns     900ns     900ns  cudaGetLastError
                    0.00%     700ns         1     700ns     700ns     700ns  cuDeviceGetName
                    0.00%     300ns         1     300ns     300ns     300ns  cuDeviceTotalMem
                    0.00%     200ns         1     200ns     200ns     200ns  cuDeviceGetUuid
```
## Descripción de resultados:
Este codigo al hacer uso de la memoria manejada por CUDA, no requiere de operaciones explicitas de copia de memoria, ya que se encarga automaticamente CUDA de mover la información a su espacio optimo y no requiere varias vueltas usando el PCI Bus. Por lo cual solo hace uso de la función para el proceso de la suma de matrices tomando un total de 12.948ms, esta función utiliza los punteros anteriormente mencionados como atributos de la función.
Dentro de las llamadas a la API, se muestra el uso de la nueva función 'cudaMallocManaged' que se encarga de asignar memoria que será utilizada por el sistema de memoria unificada y devuelve el puntero a esa memoria, este tipo de asignación de memoria se libera de manera normal, utilizando 'cudaFree' que en este caso abarcó solo 3.31% del tiempo, seguida por 'cudaDeviceSynchronize' que espera a que el dispositivo termine su tarea con 1.52% del tiempo total en 1 llamada y 'cudaLaunchKernel' que solo ejecuta una función del dispositivo.
El codigo tambien hace las llamadas normales de 'cuDeviceGetAttribute' para recibir los atributos especificos del dispositivo a usar, 'cudaSetDevice' para definir el dispositivo a usar, 'cudaGetDeviceProperties' para determinar las propiedades del dispositivo, 'cuDeviceGetCount' para establecer el numero de dispositivos disponibles, 'cuDeviceGet' que regresa el identificador del dispositivo, 'cuDeviceGetName' que regresa el nombre del dispositivo ingresado, 'cuDeviceTotalMem' que regresa la capacidad de memoria total del dispositivo, 'cuDeviceGetUuid' regresa el identificador unico universal del dispositivo ingresado.

# sumMatrixGPUManual.cu
Este codigo ejecuta el mismo proceso de suma de matrices que el anterior, con la diferencia de que este lo hace a traves del uso de llamadas de copia de memoria explicitas.
```
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   65.52%  27.101ms         2  13.550ms  8.3698ms  18.731ms  [CUDA memcpy HtoD]
                   30.63%  12.669ms         1  12.669ms  12.669ms  12.669ms  [CUDA memcpy DtoH]
                    2.69%  1.1118ms         2  555.89us  288.73us  823.04us  sumMatrixGPU(float*, float*, float*, int, int)
                    1.16%  479.42us         2  239.71us  238.91us  240.51us  [CUDA memset]
      API calls:   87.57%  607.17ms         3  202.39ms  713.10us  605.72ms  cudaMalloc
                    6.50%  45.038ms         3  15.013ms  8.6183ms  23.545ms  cudaMemcpy
                    5.26%  36.474ms         1  36.474ms  36.474ms  36.474ms  cudaDeviceReset
                    0.33%  2.2576ms         1  2.2576ms  2.2576ms  2.2576ms  cuDeviceGetPCIBusId
                    0.19%  1.3256ms         3  441.87us  223.90us  799.30us  cudaFree
                    0.13%  929.30us         1  929.30us  929.30us  929.30us  cudaDeviceSynchronize
                    0.01%  62.700us         2  31.350us  24.300us  38.400us  cudaMemset
                    0.01%  62.500us         2  31.250us  28.200us  34.300us  cudaLaunchKernel
                    0.00%  15.600us       101     154ns     100ns  1.0000us  cuDeviceGetAttribute
                    0.00%  7.3000us         1  7.3000us  7.3000us  7.3000us  cudaSetDevice
                    0.00%  7.1000us         1  7.1000us  7.1000us  7.1000us  cudaGetDeviceProperties
                    0.00%  1.3000us         3     433ns     200ns     900ns  cuDeviceGetCount
                    0.00%  1.1000us         1  1.1000us  1.1000us  1.1000us  cuDeviceGetName
                    0.00%  1.0000us         2     500ns     200ns     800ns  cuDeviceGet
                    0.00%     700ns         1     700ns     700ns     700ns  cudaGetLastError
                    0.00%     300ns         1     300ns     300ns     300ns  cuDeviceTotalMem
                    0.00%     200ns         1     200ns     200ns     200ns  cuDeviceGetUuid
```
## Descripción de Resultados:
Este codigo hace uso de la copia de memoria del host al dispositivo 2 veces que utiliza un total 65.52% del tiempo, seguido por una copia de memoria del dispositivo al host con 30.63% del tiempo total. Despues de estas copias de memorias el tiempo restante se divide en la funcion 'sumMatrixGPU' que se encarga de sumar las matrices ingresadas y una funcion nueva de cuda 'cudaMemset' que se encarga de asignar valores especificos a un espacio de memoria, ingresando el puntero a la memoria y el valor a establecer.
Como es costumbre la llamada a API con mas tiempo abarcado fue 'cudaMalloc' para la asignación de memoria e inicialización de subsistema con 87.57% del tiempo la cual se llamó un total de 3 veces, seguida por las operaciones de copia de memoria (cudaMemcpy) con un 6.50% de uso en 3 llamadas, 'cudaDeviceReset' para reestablecer el dispositivo (5.26%, 1 llamada), 'cuDeviceGetPCIBusId' para obtener el ID del PCI Bus del dispositivo, 'cudaFree' para liberación de memoria (0.19%, 3 llamadas), 'cuDeviceSynchronize' para la espera de finalización de tareas (0.13%, 1 llamada), la antes mencionada 'cudaMemset' para establecer los valores de puntos de memoria especificos (0.01%, 1 llamada) y la ejecución de función del dispositivo mediante 'cudaLaunchKernel' (0.01%, 1 llamada).
El codigo tambien hace las llamadas normales de 'cuDeviceGetAttribute' para recibir los atributos especificos del dispositivo a usar, 'cudaSetDevice' para definir el dispositivo a usar, 'cudaGetDeviceProperties' para determinar las propiedades del dispositivo, 'cuDeviceGetCount' para establecer el numero de dispositivos disponibles, 'cuDeviceGet' que regresa el identificador del dispositivo, 'cuDeviceGetName' que regresa el nombre del dispositivo ingresado, 'cuDeviceTotalMem' que regresa la capacidad de memoria total del dispositivo, 'cuDeviceGetUuid' regresa el identificador unico universal del dispositivo ingresado.

# readSegment.cu
Opuesto al primer codigo realizado (writeSegment.cu), este ejemplo muestra el impacto de lecturas desalineadas en un valor flotante.
```
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   49.71%  992.10us         1  992.10us  992.10us  992.10us  [CUDA memcpy DtoH]
                   45.41%  906.47us         2  453.23us  447.23us  459.23us  [CUDA memcpy HtoD]
                    2.48%  49.408us         1  49.408us  49.408us  49.408us  readOffset(float*, float*, float*, int, int)
                    2.40%  48.001us         1  48.001us  48.001us  48.001us  warmup(float*, float*, float*, int, int)
      API calls:   93.88%  603.77ms         3  201.26ms  313.00us  603.14ms  cudaMalloc
                    5.02%  32.299ms         1  32.299ms  32.299ms  32.299ms  cudaDeviceReset
                    0.52%  3.3638ms         3  1.1213ms  585.30us  2.1168ms  cudaMemcpy
                    0.40%  2.5464ms         1  2.5464ms  2.5464ms  2.5464ms  cuDeviceGetPCIBusId
                    0.13%  833.20us         3  277.73us  167.00us  455.50us  cudaFree
                    0.03%  206.30us         2  103.15us  68.900us  137.40us  cudaDeviceSynchronize
                    0.01%  65.800us         2  32.900us  16.900us  48.900us  cudaLaunchKernel
                    0.00%  15.800us       101     156ns     100ns  1.4000us  cuDeviceGetAttribute
                    0.00%  5.5000us         1  5.5000us  5.5000us  5.5000us  cudaSetDevice
                    0.00%  4.9000us         1  4.9000us  4.9000us  4.9000us  cudaGetDeviceProperties
                    0.00%  1.2000us         2     600ns     600ns     600ns  cudaGetLastError
                    0.00%     900ns         3     300ns     100ns     600ns  cuDeviceGetCount
                    0.00%     800ns         2     400ns     100ns     700ns  cuDeviceGet
                    0.00%     800ns         1     800ns     800ns     800ns  cuDeviceGetName
                    0.00%     500ns         1     500ns     500ns     500ns  cuDeviceTotalMem
                    0.00%     200ns         1     200ns     200ns     200ns  cuDeviceGetUuid
```
## Descripción de resultados:
Este codigo toma la mayoria de tiempo en las operaciones de copia de memoria, usando 48.71% en 1 copia del dispositivo al host y 45.41% en 2 copias del host al dispositivo. Seguidas por las funciones del proceso del codigo, 'readOffset' que se encarga de realizar las operaciones en arreglos y las almacena en un tercero, usando un 2.48% del tiempo y 'warmup' que repite la operación anterior usando un 2.40% del tiempo.
Utiliza las llamadas comunes a la API CUDA, como la asignación de memoria con 'cudaMalloc' (93.88%, 3 llamadas), reestablecimiento de dispositivo con 'cudaDeviceReset' (5.02%, 1 llamda), las operaciones de copia mencionadas anteriormente con 'cudaMemcpy' (0.52%, 3 llamadas), la obtencion del PCI bus con 'cuDevicePCIBusId' (0.40%, 1 llamada), liberación de memoria con 'cudaFree' (0.13%, 3 llamadas), sincronización de dispositivos con 'cudaDeviceSynchronize' (0.03%, 2 llamadas) y la ejecución de funciones con 'cudaLaunchKernel' (0.01%, 2 llamadas).
El codigo tambien hace las llamadas normales de 'cuDeviceGetAttribute' para recibir los atributos especificos del dispositivo a usar, 'cudaSetDevice' para definir el dispositivo a usar, 'cudaGetDeviceProperties' para determinar las propiedades del dispositivo, 'cuDeviceGetCount' para establecer el numero de dispositivos disponibles, 'cuDeviceGet' que regresa el identificador del dispositivo, 'cuDeviceGetName' que regresa el nombre del dispositivo ingresado, 'cuDeviceTotalMem' que regresa la capacidad de memoria total del dispositivo, 'cuDeviceGetUuid' regresa el identificador unico universal del dispositivo ingresado.

# readSegmentUnroll.cu
Similar al codigo pasado, estudia el impacto de lecturas desalineadas en operaciones con valores flotantes con la diferencia de usar funciones que reducen el impacto de estas mediante 'unrolling' que es una tecnica de optimizacion que mejora la velocidad de los bucles a costa de aumentar su tamaño en memoria.
```
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   64.13%  2.0672ms         3  689.07us  470.56us  864.49us  [CUDA memcpy DtoH]
                   27.79%  895.65us         2  447.83us  446.53us  449.12us  [CUDA memcpy HtoD]
                    1.94%  62.593us         4  15.648us  15.360us  16.320us  [CUDA memset]
                    1.56%  50.368us         1  50.368us  50.368us  50.368us  readOffsetUnroll4(float*, float*, float*, int, int)
                    1.55%  49.984us         1  49.984us  49.984us  49.984us  readOffset(float*, float*, float*, int, int)
                    1.54%  49.632us         1  49.632us  49.632us  49.632us  readOffsetUnroll2(float*, float*, float*, int, int)
                    1.49%  47.904us         1  47.904us  47.904us  47.904us  warmup(float*, float*, float*, int, int)
      API calls:   93.30%  592.46ms         3  197.49ms  309.10us  591.77ms  cudaMalloc
                    5.46%  34.676ms         1  34.676ms  34.676ms  34.676ms  cudaDeviceReset
                    0.69%  4.4052ms         5  881.04us  498.20us  1.8633ms  cudaMemcpy
                    0.32%  2.0617ms         1  2.0617ms  2.0617ms  2.0617ms  cuDeviceGetPCIBusId
                    0.12%  749.60us         3  249.87us  170.00us  390.60us  cudaFree
                    0.06%  357.30us         4  89.325us  71.700us  130.70us  cudaDeviceSynchronize
                    0.02%  144.90us         4  36.225us  22.500us  52.700us  cudaMemset
                    0.01%  91.300us         4  22.825us  9.4000us  47.700us  cudaLaunchKernel
                    0.00%  14.600us       101     144ns     100ns  1.3000us  cuDeviceGetAttribute
                    0.00%  6.9000us         1  6.9000us  6.9000us  6.9000us  cudaGetDeviceProperties
                    0.00%  5.7000us         1  5.7000us  5.7000us  5.7000us  cudaSetDevice
                    0.00%  2.4000us         4     600ns     500ns     700ns  cudaGetLastError
                    0.00%  1.5000us         2     750ns     300ns  1.2000us  cuDeviceGet
                    0.00%  1.2000us         3     400ns     100ns     900ns  cuDeviceGetCount
                    0.00%     800ns         1     800ns     800ns     800ns  cuDeviceGetName
                    0.00%     400ns         1     400ns     400ns     400ns  cuDeviceTotalMem
                    0.00%     200ns         1     200ns     200ns     200ns  cuDeviceGetUuid
```
## Descripción de resultados:
Este codigo toma la mayoria de tiempo en las operaciones de copia de memoria, usando 64.13% en 3 copias del dispositivo al host y 27.79% en 2 copias del host al dispositivo, además de establecer valores en memoria con 'cudaMemset' usando un 1.94% del tiempo total. Seguidas por las funciones del proceso del codigo, 'readOffsetUnroll4' que se encarga de realizar las operaciones en arreglos y las almacena en un tercero con la diferencia de dividir esta suma 4 veces, usando un 1.56% del tiempo, 'readOffset' que hace esta operación de manera convencional usando 1.55% del tiempo, muy similar a la siguiente 'readOffsetUnroll2 que solo segmenta la suma 2 veces y toma un 1.54%, una diferencia de solo 0.352 microsegundose entre ambas. Todo esto seguido de 'warmup' que usa un 1.49% del tiempo.
Utiliza las llamadas comunes a la API CUDA, como la asignación de memoria con 'cudaMalloc' (93.88%, 3 llamadas), reestablecimiento de dispositivo con 'cudaDeviceReset' (5.02%, 1 llamda), las operaciones de copia mencionadas anteriormente con 'cudaMemcpy' (0.52%, 3 llamadas), la obtencion del PCI bus con 'cuDevicePCIBusId' (0.40%, 1 llamada), liberación de memoria con 'cudaFree' (0.13%, 3 llamadas), sincronización de dispositivos con 'cudaDeviceSynchronize' (0.03%, 2 llamadas), establecimiento de valores en memoria con 'cudaMemset' (0.02%, 4 llamadas) y la ejecución de funciones en dispositivo con 'cudaLaunchKernel' (0.01%, 4 llamadas).
El codigo tambien hace las llamadas normales de 'cuDeviceGetAttribute' para recibir los atributos especificos del dispositivo a usar, 'cudaSetDevice' para definir el dispositivo a usar, 'cudaGetDeviceProperties' para determinar las propiedades del dispositivo, 'cuDeviceGetCount' para establecer el numero de dispositivos disponibles, 'cuDeviceGet' que regresa el identificador del dispositivo, 'cuDeviceGetName' que regresa el nombre del dispositivo ingresado, 'cuDeviceTotalMem' que regresa la capacidad de memoria total del dispositivo, 'cuDeviceGetUuid' regresa el identificador unico universal del dispositivo ingresado.
