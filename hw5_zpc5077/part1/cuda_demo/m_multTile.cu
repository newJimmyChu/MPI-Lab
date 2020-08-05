#include <stdio.h>
#include <sys/time.h>

#define N 512
#define TILE_WIDTH 16

__global__ void matrixMult (int *a, int *b, int *c, int width);
void matrixMultCPU (int a[N][N], int b[N][N], int c[N][N], int width);

double myDiffTime(struct timeval &start, struct timeval &end)
{
	double d_start, d_end;
	d_start = (double)(start.tv_sec + start.tv_usec/1000000.0);
	d_end = (double)(end.tv_sec + end.tv_usec/1000000.0);
	return (d_end - d_start);
}

int main() 
{
	int a[N][N], b[N][N], c[N][N], g[N][N];
	timeval start, end;

	int *dev_a, *dev_b, *dev_c;
	int size = N * N * sizeof(int);

	// initialize matrices a and b with appropriate values
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			a[i][j] = i*N + j;
			b[i][j] = i + j;
		}
	}

	// initialize a and b matrices here
	cudaMalloc((void **) &dev_a, size);
	cudaMalloc((void **) &dev_b, size);
	cudaMalloc((void **) &dev_c, size);

	gettimeofday(&start, NULL);

	cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, size, cudaMemcpyHostToDevice);

	dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
	dim3 dimGrid((int)ceil(N/dimBlock.x), (int)ceil(N/dimBlock.y));

	matrixMult<<<dimGrid, dimBlock>>>(dev_a, dev_b, dev_c, N);
	cudaDeviceSynchronize();

	cudaMemcpy(g, dev_c, size, cudaMemcpyDeviceToHost);

	gettimeofday(&end, NULL);
	printf("GPU Time for %i additions: %f\n", N, myDiffTime(start, end));

	gettimeofday(&start, NULL);
	matrixMultCPU(a, b, c, N);
	
	gettimeofday(&end, NULL);
	printf("CPU Time for %i additions: %f\n", N, myDiffTime(start, end));

	cudaFree(dev_a); cudaFree(dev_b); cudaFree(dev_c);

	// print verification
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			if (c[i][j] != g[i][j])
			{
				printf("Results do not match! %i, %i, c=%i, g=%i\n", i, j, c[i][j], g[i][j]);
				exit(1);
			}
		}
	}

}

__global__ void matrixMult(int* A, int* B, int* C, int width)
{
	int k, sum = 0;
	int col = blockIdx.x*TILE_WIDTH + threadIdx.x;
	int row = blockIdx.y*TILE_WIDTH + threadIdx.y;
	if(col < width && row < width) 
	{
		for (k = 0; k < width; k++)
			sum += A[row * width + k] * B[k * width + col];

		C[row * width + col] = sum;
	}
}

void matrixMultCPU (int a[N][N], int b[N][N], int c[N][N], int width) 
{
	for (int i = 0; i < width; i++) 
	{
		for (int j = 0; j < width; j++) 
		{
			int sum = 0;
			for (int k = 0; k < width; k++) 
			{
				int m = a[i][k];
				int n = b[k][j];
				sum += m * n;
			}
		c[i][j] = sum;
		}
	}
}
