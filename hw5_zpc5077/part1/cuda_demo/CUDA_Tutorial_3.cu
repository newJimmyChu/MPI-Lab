#include <stdio.h>
#include <sys/time.h>
#include <omp.h>

/*
This file can be downloaded from supercomputingblog.com.
This is part of a series of tutorials that demonstrate how to use CUDA
The tutorials will also demonstrate the speed of using CUDA
*/

// IMPORTANT NOTE: for this data size, your graphics card should have at least 256 megabytes of memory.
// If your GPU has less memory, then you will need to decrease this data size.

#define MAX_DATA_SIZE		1024*1024*32		// about 32 million elements. 
// The max data size must be an integer multiple of 128*256, because each block will have 256 threads,
// and the block grid width will be 128. These are arbitrary numbers I choose.
#define THREADS_PER_BLOCK	256
#define BLOCKS_PER_GRID_ROW 128


double myDiffTime(struct timeval &start, struct timeval &end)
{
	double d_start, d_end;
	d_start = (double)(start.tv_sec + start.tv_usec/1000000.0);
	d_end = (double)(end.tv_sec + end.tv_usec/1000000.0);
	return (d_end - d_start);
}

__global__ void getStats(float *pArray, float *pMaxResults, float *pMinResults, float *pAvgResults)
{
	// Declare arrays to be in shared memory.
	// 256 elements * (4 bytes / element) * 3 = 3KB.
	__shared__ float min[256];
	__shared__ float max[256];
	__shared__ float avg[256];

	// Calculate which element this thread reads from memory
	int arrayIndex = 128*256*blockIdx.y + 256*blockIdx.x + threadIdx.x;
	min[threadIdx.x] = max[threadIdx.x] = avg[threadIdx.x] = pArray[arrayIndex];
	__syncthreads();
	int nTotalThreads = blockDim.x;	// Total number of active threads

	while(nTotalThreads > 1)
	{
		int halfPoint = (nTotalThreads >> 1);	// divide by two
		// only the first half of the threads will be active.
		if (threadIdx.x < halfPoint)
		{
			// Get the shared value stored by another thread
			float temp = min[threadIdx.x + halfPoint];
			if (temp < min[threadIdx.x]) min[threadIdx.x] = temp;
			temp = max[threadIdx.x + halfPoint];
			if (temp > max[threadIdx.x]) max[threadIdx.x] = temp;
			
			// when calculating the average, sum and divide
			avg[threadIdx.x] += avg[threadIdx.x + halfPoint];
			avg[threadIdx.x] /= 2;
		}
		__syncthreads();

		nTotalThreads = (nTotalThreads >> 1);	// divide by two.
	}

	// At this point in time, thread zero has the min, max, and average
	// It's time for thread zero to write it's final results.
	// Note that the address structure of pResults is different, because
	// there is only one value for every thread block.

	if (threadIdx.x == 0)
	{
		pMaxResults[128*blockIdx.y + blockIdx.x] = max[0];
		pMinResults[128*blockIdx.y + blockIdx.x] = min[0];
		pAvgResults[128*blockIdx.y + blockIdx.x] = avg[0];
	}
}

void getStatsCPU(float *pArray, int nElems, float *pMin, float *pMax, float *pAvg)
{
	// This function uses the CPU to find the min, max and average of an array
	
	if (nElems <= 0) return;
	float min, max, avg;
	min = max = avg = pArray[0];
	
	for (int i=1; i < nElems; i++)
	{
		float temp = pArray[i];
		if (temp < min) min = temp;
		if (temp > max) max = temp;
		avg += temp;	// we will divide once after for loop for speed.
	}
	avg /= (float)nElems;
	*pMin = min;
	*pMax = max;
	*pAvg = avg;
}


////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv){
	float *h_data, *h_resultMax, *h_resultMin, *h_resultAvg;
	float *d_data, *d_resultMax, *d_resultMin, *d_resultAvg;
	double gpuTime;
    	int i;

	timeval start, end;


    	printf("Initializing data...\n");
	h_data     = (float *)malloc(sizeof(float) * MAX_DATA_SIZE);
	h_resultMax = (float *)malloc(sizeof(float) * MAX_DATA_SIZE / THREADS_PER_BLOCK);
	h_resultMin = (float *)malloc(sizeof(float) * MAX_DATA_SIZE / THREADS_PER_BLOCK);
	h_resultAvg = (float *)malloc(sizeof(float) * MAX_DATA_SIZE / THREADS_PER_BLOCK);

	cudaMalloc( (void **)&d_data, sizeof(float) * MAX_DATA_SIZE);
	cudaMalloc( (void **)&d_resultMax, sizeof(float) * MAX_DATA_SIZE / THREADS_PER_BLOCK);
	cudaMalloc( (void **)&d_resultMin, sizeof(float) * MAX_DATA_SIZE / THREADS_PER_BLOCK);
	cudaMalloc( (void **)&d_resultAvg, sizeof(float) * MAX_DATA_SIZE / THREADS_PER_BLOCK);


	srand(123);
	for(i = 0; i < MAX_DATA_SIZE; i++)
	{
		h_data[i] = (float)rand() / (float)RAND_MAX;
	}

	int firstRun = 1;	// Indicates if it's the first execution of the for loop
	const int useGPU = 1;	// When 0, only the CPU is used. When 1, only the GPU is used

	for (int dataAmount = MAX_DATA_SIZE; dataAmount > BLOCKS_PER_GRID_ROW*THREADS_PER_BLOCK; dataAmount /= 2)
	{
		float tempMin,tempMax,tempAvg;

		int blockGridWidth = BLOCKS_PER_GRID_ROW;
		int blockGridHeight = (dataAmount / THREADS_PER_BLOCK) / blockGridWidth;

		dim3 blockGridRows(blockGridWidth, blockGridHeight);
		dim3 threadBlockRows(THREADS_PER_BLOCK, 1);

		// Start the timer.
		// We want to measure copying data, running the kernel, and copying the results back to host
        	gettimeofday(&start, NULL);
        

		if (useGPU == 1)
		{
			// Copy the data to the device
			cudaMemcpy(d_data, h_data, sizeof(float) * dataAmount, cudaMemcpyHostToDevice);

			// Do the multiplication on the GPU
			getStats<<<blockGridRows, threadBlockRows>>>(d_data, d_resultMax, d_resultMin, d_resultAvg);

			cudaThreadSynchronize();

			// Copy the data back to the host
			cudaMemcpy(h_resultMin, d_resultMin, sizeof(float) * dataAmount / THREADS_PER_BLOCK, cudaMemcpyDeviceToHost);
			cudaMemcpy(h_resultMax, d_resultMax, sizeof(float) * dataAmount / THREADS_PER_BLOCK, cudaMemcpyDeviceToHost);
			cudaMemcpy(h_resultAvg, d_resultAvg, sizeof(float) * dataAmount / THREADS_PER_BLOCK, cudaMemcpyDeviceToHost);

			// Each block returned one result, so lets finish this off with the cpu.
			// By using CUDA, we basically reduced how much the CPU would have to work by about 256 times.
			
			tempMin = h_resultMin[0];
			tempMax = h_resultMax[0];
			tempAvg = h_resultAvg[0];
			for (int i=1 ; i < dataAmount / THREADS_PER_BLOCK; i++)
			{
				if (h_resultMin[i] < tempMin) tempMin = h_resultMin[i];
				if (h_resultMax[i] > tempMax) tempMax = h_resultMax[i];
				tempAvg += h_resultAvg[i];
			}
			tempAvg /= (dataAmount / THREADS_PER_BLOCK);
		}
		else
		{
			// We're using the CPU only
			getStatsCPU(h_data, dataAmount, &tempMin, &tempMax, &tempAvg);
		}
		printf("Min: %f Max %f Avg %f\n", tempMin, tempMax, tempAvg);

		// Stop the timer, print the total round trip execution time.
		gettimeofday(&end, NULL);
		gpuTime = myDiffTime(start, end);
		if (!firstRun || !useGPU)
		{
			printf("Elements: %d - convolution time : %f msec - %f Multiplications/sec\n", dataAmount, gpuTime, blockGridHeight * 128 * 256 / (gpuTime * 0.001));
		}
		else
		{
			firstRun = 0;
			// We discard the results of the first run because of the extra overhead incurred
			// during the first time a kernel is ever executed.
			dataAmount *= 2;	// reset to first run value
		}
	}

    printf("Cleaning up...\n");
	cudaFree(d_resultMin );
	cudaFree(d_resultMax );
	cudaFree(d_resultAvg );
	cudaFree(d_data);
	free(h_resultMin);
	free(h_resultMax);
	free(h_resultAvg);
	free(h_data);


}
