#include "floyd_warshall.cuh"

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <limits.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include <cuda_runtime_api.h>
#include <memory>


#define min(a,b) (((a)<(b))?(a):(b))
#define BLOCK_SIZE 16

static size_t data_size = 10;

//state variable
static uint64_t rand_state;

__global__ void parallel_floyd_warshall(int* graph, int n, int* path)
{
	// block indices
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int aBegin = n * BLOCK_SIZE * by;
	int aEnd = aBegin + n - 1;
	int aStep = BLOCK_SIZE;

	int bBegin = BLOCK_SIZE * bx;
	int bStep = BLOCK_SIZE * n;

	int pathsub = 0;

	for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
		//load block into shared memory
		__shared__ int graph_s[BLOCK_SIZE][BLOCK_SIZE];
		__shared__ int path_s[BLOCK_SIZE][BLOCK_SIZE];
		graph_s[ty][tx] = graph[a + n * ty + tx];
		path_s[ty][tx] = path[b + n * ty + tx];
		__syncthreads();

		//find minimum for block
		for (int k = 0; k < BLOCK_SIZE; ++k) {
			pathsub = graph_s[ty][k] < graph_s[ty][k] + path_s[k][tx] ?
				graph_s[ty][k] : graph_s[ty][k] + path_s[k][tx];
		}
		__syncthreads();
	}
	//writeback
	int pathwrite = n * BLOCK_SIZE * by + BLOCK_SIZE * bx;
	path[pathwrite + n * ty + tx] = pathsub;

}


int main()
{
	//create graph
	//int* graph = (int*)calloc(sizeof(int), n * n);
	//for (int i = 0; i < n; i++) {
	//	for (int j = 0; j < n; j++) {
	//		if (i == j) {
	//			graph[(i * n) + j] = 0;
	//		}
	//		else {
	//			graph[(i * n) + j] = xrand();
	//		}
	//	}
	//}

	//density will be between 0 and 100, indication the % of number of directed edges in graph
	//range will be the range of edge weighting of directed edges
	
		int N = 700;
		int density = 25;
		int Prange = (100 / density);
		int* G = (int*)calloc(sizeof(int), N * N);
		int range = 1000;
		int INF = std::numeric_limits<int>::max();
			for (int i = 0; i < N; i++) {
			for (int j = 0; j < N; j++) {
				if (i == j) {//set G[i][i]=0
					G[i * N + j] = 0;
					continue;
				}
				int pr = rand() % Prange;
				G[i * N + j] = pr == 0 ? ((rand() % range) + 1) : INF;
				//set edge random edge weight to random value, or to INF
			}
		}
	

	int* path = NULL;

	
	int* graph_d;
	cudaMalloc(&graph_d, N * N);
	cudaMemcpy(graph_d, G, N * N, cudaMemcpyHostToDevice);

	int* path_d;
	cudaMalloc(&path_d, N * N);

	int grid = 1;
	int block = 1;

	parallel_floyd_warshall <<<grid, block >>> (graph_d, N, path_d);


	//free memory
	cudaFree(path_d);
	cudaFree(graph_d);
	free(path);
	free(G);
	return 0;
}