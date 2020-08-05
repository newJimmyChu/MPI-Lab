#include <stdio.h>
#include <sys/time.h>

#define N 65535
#define T 1024 // max threads per block

double myDiffTime(struct timeval &start, struct timeval &end)
{
	double d_start, d_end;
	d_start = (double)(start.tv_sec + start.tv_usec/1000000.0);
	d_end = (double)(end.tv_sec + end.tv_usec/1000000.0);
	return (d_end - d_start);
}

__global__ void vecAdd (int *a, int *b, int *c);
void vecAddCPU(int *a, int *b, int *c);

int main() 
{
	int a[N], b[N], c[N];
	int *dev_a, *dev_b, *dev_c;
	timeval start, end;
	// initialize a and b with real values 
	for (int i = 0; i < N; i++)
	{
		a[i] = i;
		b[i] = N-i;
		c[i] = 0;
	}

	int size = N * sizeof(int);
	cudaMalloc((void**)&dev_a, size);
	cudaMalloc((void**)&dev_b, size);
	cudaMalloc((void**)&dev_c, size);
	

	gettimeofday(&start, NULL);
	cudaMemcpy(dev_a, a, size,cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, size,cudaMemcpyHostToDevice);
	//gettimeofday(&start, NULL);
	vecAdd<<<(int)ceil(N/T),T>>>(dev_a,dev_b,dev_c);
	//gettimeofday(&end, NULL);	

	cudaMemcpy(c, dev_c, size,cudaMemcpyDeviceToHost);
	gettimeofday(&end, NULL);

	printf("GPU Time for %i additions: %f\n", N, myDiffTime(start, end));

	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);

	gettimeofday(&start, NULL);
	vecAddCPU(a, b, c);
	gettimeofday(&end, NULL);	

	printf("CPU Time for %i additions: %f\n", N, myDiffTime(start, end));

	exit (0);
}

__global__ void vecAdd (int *a, int *b, int *c) 
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N) 
	{
		c[i] = a[i] + b[i];
	}
}

void vecAddCPU(int *a, int *b, int *c)
{
	for (int i = 0; i < N; i++)
		c[i] = a[i] + b[i];
}
