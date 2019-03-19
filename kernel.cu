
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <fstream>
#include <stdio.h>
#include <iostream>



cudaError_t thomasWithCuda(double *Zs, double *As, double *Bs, double *Cs, double *Ys, int *lengths, int *starts,  int N, int size);



__global__ void thomasKernel(double *Zs, double *As, double *Bs, double *Cs, double *Ys, int *lengths, int *starts){

	int t_idx = threadIdx.x;
	int M = lengths[t_idx];
	double *a = &As[starts[t_idx] - t_idx];
	double *b = &Bs[starts[t_idx]];
	double *c = &Cs[starts[t_idx] - t_idx];
	double *y = &Ys[starts[t_idx]];
	double *z = &Zs[starts[t_idx]];

	c[0] = c[0] / b[0];
	y[0] = y[0] / b[0];


	for (int i = 1; i < M - 1; i++){
		c[i] = c[i] / (b[i] - a[i - 1] * c[i - 1]);	
	}

	for (int i = 1; i < M ; i++){
		y[i] = (y[i] - a[i - 1] * y[i - 1]) / (b[i] - a[i - 1] * c[i - 1]);
	}
	
		z[M - 1] = y[M - 1];

	for (int i = M - 2; i > -1; i--){
		z[i] = y[i] - c[i] * z[i + 1];

	}

}

double* extract_udiag(double ** matrix, int M){

	double *arr = (double*)malloc((M - 1) * sizeof(double));
	for (int i = 1; i < M; i++){
		arr[i - 1] = matrix[i][i - 1]; 
	}
	return arr;
}

double* extract_diag(double ** matrix, int M){

	double *arr = (double*)malloc((M) * sizeof(double));
	for (int i = 0; i < M; i++){
		arr[i] = matrix[i][i]; 
	}
	return arr;
}

double* extract_adiag(double ** matrix, int M){

	double *arr = (double*)malloc((M - 1) * sizeof(double));
	for (int i = 1; i < M; i++){
		arr[i - 1] = matrix[i - 1][i]; 
	}
	return arr;
}


int main()
{
	std::ifstream inFile;
	inFile.open("C:/Users/Sergey/Documents/visual studio 2012/Projects/trial-cuda/systems.txt");
	int N;
	int M;

	inFile >> N; // number of systems
	double ***Systems = (double***)malloc(N * sizeof(double**));
	int *lengths = (int*)malloc(N * sizeof(int));
	double **Solutions = (double**)malloc(N * sizeof(double*)); 


	for(int i = 0; i < N; i++){
		inFile >> M;

		lengths[i] = M;
		Systems[i] = (double**)malloc(M * sizeof(double *));

		for (int j = 0; j < M; j++){
			Systems[i][j] = (double*)malloc(M * sizeof(double));
			for (int k = 0; k < M; k++){
				inFile >> Systems[i][j][k];
			}
		}

		Solutions[i] = (double*)malloc(M * sizeof(double));
		for (int j = 0; j < M; j++){
			inFile >> Solutions[i][j];
		}
	}

	inFile.close();
	int *starts = (int*)malloc(N * sizeof(int));

	int count = 0;
	for (int i = 0; i < N; i++){
		starts[i] = count;
		count += lengths[i];
		
	}

	//EXTRACTING DIAGONALS
	double *As = (double*)malloc((count - N) * sizeof(double));
	double *Bs = (double*)malloc(count * sizeof(double));
	double *Cs = (double*)malloc((count - N) * sizeof(double));
	double *Ys = (double*)malloc(count * sizeof(double));
	double *Zs = (double*)malloc(count * sizeof(double));

	double *extract1, *extract2, *extract3;
	for (int i = 0; i < N; i++){
		extract1 = extract_udiag(Systems[i], lengths[i]);
		extract2 = extract_diag(Systems[i], lengths[i]);
		extract3 = extract_adiag(Systems[i], lengths[i]);

		for (int j = starts[i]; j < starts[i] + lengths[i]; j++){
			Bs[j] = extract2[j - starts[i]];
			Ys[j] = Solutions[i][j - starts[i]];
			
		}

		for (int j = starts[i] - i; j < starts[i] + lengths[i] - (i + 1); j++){
			As[j] = extract1[j - (starts[i] - i)];
			Cs[j] = extract3[j - (starts[i] - i)];

		}

	}
	
	cudaError_t cudaStatus = thomasWithCuda(Zs, As, Bs, Cs, Ys, lengths, starts, N, count);
	cudaStatus = cudaDeviceReset();

	for (int i = 0; i < N; i++){
		for (int j = starts[i]; j < starts[i] + lengths[i]; j++)
			std::cout << Zs[j] << ' ';
		std::cout << '\n';
	}
		


	free(As);
	free(Bs);
	free(Cs);
	free(Zs);
	free(Ys);
	free(starts);
	free(Systems);
	free(lengths);
	free(Solutions);



    return 0;
}

cudaError_t thomasWithCuda(double *Zs, double *As, double *Bs, double *Cs, double *Ys, int *lengths, int *starts,  int N, int size)
{
	double *As_c;
	double *Bs_c;
	double *Cs_c;
	double *Zs_c;
	double *Ys_c;
	int *lengths_c;
	int *starts_c;


	cudaError_t cudaStatus;

	cudaStatus = cudaSetDevice(1);

	cudaStatus = cudaMalloc((void**)&As_c, (size - N) * sizeof(double));
	cudaStatus = cudaMalloc((void**)&Bs_c, size * sizeof(double));
	cudaStatus = cudaMalloc((void**)&Cs_c, (size - N) * sizeof(double));
	cudaStatus = cudaMalloc((void**)&Zs_c, size * sizeof(double));
	cudaStatus = cudaMalloc((void**)&Ys_c, size * sizeof(double));
	cudaStatus = cudaMalloc((void**)&lengths_c, N * sizeof(int));
	cudaStatus = cudaMalloc((void**)&starts_c, N * sizeof(int));


	cudaStatus = cudaMemcpy(As_c, As, (size - N) * sizeof(double), cudaMemcpyHostToDevice);
	cudaStatus = cudaMemcpy(Bs_c, Bs, size * sizeof(double), cudaMemcpyHostToDevice);
	cudaStatus = cudaMemcpy(Cs_c, Cs, (size - N) * sizeof(double), cudaMemcpyHostToDevice);
	cudaStatus = cudaMemcpy(Ys_c, Ys, size * sizeof(double), cudaMemcpyHostToDevice);
	cudaStatus = cudaMemcpy(lengths_c, lengths, N * sizeof(int), cudaMemcpyHostToDevice);
	cudaStatus = cudaMemcpy(starts_c, starts, N * sizeof(int), cudaMemcpyHostToDevice);

	thomasKernel<<<1, N>>>(Zs_c, As_c, Bs_c, Cs_c, Ys_c, lengths_c, starts_c);

	cudaStatus = cudaDeviceSynchronize();

	cudaStatus = cudaMemcpy(Zs, Zs_c, size * sizeof(double), cudaMemcpyDeviceToHost);

	cudaFree(As_c);
	cudaFree(Bs_c);
	cudaFree(Cs_c);
	cudaFree(Zs_c);
	cudaFree(Ys_c);
	cudaFree(lengths_c);
	cudaFree(starts_c);

	return cudaStatus;
}

