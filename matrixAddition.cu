#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <helper_cuda.h>
#include <helper_timer.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

struct Matrix {
	int rows;
	int cols;
	float* data;
};

Matrix* createMatrix(int rows, int columns) {
	int size = rows * columns;

	Matrix* m = (Matrix*) malloc(sizeof(Matrix));
	m->rows = rows;
	m->cols = columns;
	m->data = (float*) malloc(size * sizeof(float));

	return m;
}

Matrix* createRandomMatrix(int rows, int columns) {
	Matrix* m = createMatrix(rows, columns);

	for (int i = 0; i < rows * columns; i++) {
		m->data[i] = ((float)rand() / (float)(RAND_MAX)) * 5;
	}

	return m;
}

void clean(Matrix* m) {
	free(m->data);
	free(m);
}

void printMatrix(Matrix* m) {
	for (int i = 0; i < m->rows; i++) {
		for (int j = 0; j < m->cols; j++) {
			printf("%.2f ", m->data[i * m->cols + j]);
		}
		printf("\n");
	}
	printf("\n");
}

Matrix* addOnCPU(Matrix* m1, Matrix* m2, float* time) {
	if (m1->rows != m2->rows || m1->cols != m2->cols) {
		return NULL;
	}

	int size = m1->rows * m1->cols;
	Matrix* m3 = createMatrix(m1->rows, m1->cols);

	StopWatchInterface* timer = NULL;
	sdkCreateTimer(&timer);
	float start = sdkGetTimerValue(&timer);
	sdkStartTimer(&timer);

	for (int i = 0; i < size; i++) {
		m3->data[i] = m1->data[i] + m2->data[i];
	}

	sdkStopTimer(&timer);
	*time = sdkGetTimerValue(&timer) - start;

	return m3;
}

__global__ void kernel(float* m1, float* m2, float* m3, const int width, const int total) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	int index = row * width + col;

	if (index < total) {
		m3[index] = m1[index] + m2[index];
	}
}

Matrix* addOnGPU(Matrix* m1, Matrix* m2, dim3 grid, dim3 block, float* time) {
	if (m1->rows != m2->rows || m1->cols != m2->cols) {
		return NULL;
	}

	int size = m1->rows * m1->cols * sizeof(float);
	float* d_m1;
	float* d_m2;
	float* d_m3;

	cudaMalloc((void**)&d_m1, size);
	cudaMemcpy(d_m1, m1->data, size, cudaMemcpyHostToDevice);
	cudaMalloc((void**)&d_m2, size);
	cudaMemcpy(d_m2, m2->data, size, cudaMemcpyHostToDevice);
	cudaMalloc((void**)&d_m3, size);

	StopWatchInterface* timer = NULL;
	sdkCreateTimer(&timer);
	float start = sdkGetTimerValue(&timer);
	sdkStartTimer(&timer);

	kernel <<<grid, block>>> (d_m1, d_m2, d_m3, m1->cols, m1->rows * m1->cols);

	checkCudaErrors(cudaThreadSynchronize());

	sdkStopTimer(&timer);
	*time = sdkGetTimerValue(&timer) - start;

	Matrix* m3 = createMatrix(m1->rows, m2->cols);

	cudaMemcpy(m3->data, d_m3, size, cudaMemcpyDeviceToHost);
	cudaFree(d_m1);
	cudaFree(d_m2);
	cudaFree(d_m3);

	return m3;
}

bool checkResult(Matrix* hostMatrix, Matrix* deviceMatrix) {
	double e = 1.0E-8;
	int size = hostMatrix->rows * hostMatrix->cols;

	if (hostMatrix->rows != deviceMatrix->rows || hostMatrix->cols != deviceMatrix->cols) {
		return false;
	}

	for (int i = 0; i < size; i++) {
		if (abs((long)(hostMatrix->data[i] - deviceMatrix->data[i])) > e) {
			printf("Matrices do not match!\n");
			printf("Index %d \n", i);
			printf("CPU: %.2f, GPU: %.2f \n", hostMatrix->data[i], deviceMatrix->data[i]);
			return false;
		}
	}

	return true;
}

#define BLOCK_WIDTH_X 16
#define BLOCK_WIDTH_Y 16

int main() {
	const int dimX = 100;
	const int dimY = 100;

	dim3 dimBlock(BLOCK_WIDTH_X, BLOCK_WIDTH_Y);
	dim3 dimGrid(ceil(dimX / (float)BLOCK_WIDTH_X), ceil(dimY / (float)BLOCK_WIDTH_Y));

	Matrix* m1 = createRandomMatrix(dimX, dimY);
	Matrix* m2 = createRandomMatrix(dimX, dimY);

	float timeCPU, timeGPU;

	Matrix* m3 = addOnCPU(m1, m2, &timeCPU);
	Matrix* m4 = addOnGPU(m1, m2, dimGrid, dimBlock, &timeGPU);

	bool same = checkResult(m3, m4);

	// printMatrix(m1);
	// printMatrix(m2);
	// printMatrix(m3);
	// printMatrix(m4);

	clean(m1);
	clean(m2);
	clean(m3);
	clean(m4);

	cudaDeviceReset();

	if (!same) {
		return 1;
	}

	printf("Execution time:\n");
	printf("CPU: %.6f ms\n", timeCPU);
	printf("GPU: %.6f ms\n", timeGPU);

	return 0;
}
