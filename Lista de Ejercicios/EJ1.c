// ConsoleApplication1.cpp : Este archivo contiene la función "main". La ejecución del programa comienza y termina ahí.
//

//Suma de Matrices
#include <stdio.h>
#include <stdlib.h>

//Matrizes
int matA[10][10];
int matB[10][10];
int matR[10][10];

int main()
{
	//Inicializar Matriz
	for (int i = 0; i < 10; i++)
	{
		for (int j = 0; j < 10; j++)
		{
			matA[i][j] = rand() % 10 + 1;
			matB[i][j] = rand() % 10 + 1;
		}
	}

	for (int i = 0; i < 10; i++)
	{
		for (int j = 0; j < 10; j++)
		{
			matR[i][j] = 0; //Inicializar valor de matriz

			for (int k = 0; k < 10; k++)
			{
				//Sumatoria de valores multiplicados
				matR[i][j] += matA[i][k] * matB[k][j];
			}
		}
	}

	//Imprimir resultado matriz
	for (int i = 0; i < 10; i++) 
	{
		for (int j = 0; j < 10; j++)
		{
			printf("Valor en [%d][%d]: %d \n", i, j, matR[i][j]);
		}
	}
}