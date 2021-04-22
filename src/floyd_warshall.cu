#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <memory>

#include "floyd_warshall.cuh"

__global__ void ParallelFloydWarshall(int* graph, int n, int* path) {
	const int block_size = 16;
	// block indices
	const int bx = blockIdx.x;
	const int by = blockIdx.y;
	const int tx = threadIdx.x;
	const int ty = threadIdx.y;

	const int a_begin = n * block_size * by;
	const int a_end = a_begin + n - 1;
	const int a_step = block_size;

	const int b_begin = block_size * bx;
	const int b_step = block_size * n;

	int pathsub = 0;

	for (int a = a_begin, b = b_begin; a <= a_end; a += a_step, b += b_step) {
		// Load block into shared memory.
		int graph_s[block_size][block_size];
		int path_s[block_size][block_size];
		graph_s[ty][tx] = graph[a + n * ty + tx];
		path_s[ty][tx] = path[b + n * ty + tx];
		__syncthreads();

		// Find minimum for block.
		for (int k = 0; k < block_size; ++k) {
			pathsub = graph_s[ty][k] < graph_s[ty][k] + path_s[k][tx] ?
				graph_s[ty][k] : graph_s[ty][k] + path_s[k][tx];
		}
		__syncthreads();
	}
	// Writeback.
	int pathwrite = n * block_size * by + block_size * bx;
	path[pathwrite + n * ty + tx] = pathsub;
}


int main()
{
	// Create graph density will be between 0 and 100,
	// indication the % of number of directed edges in graph
	// range will be the range of edge weighting of directed edges.

	const int n = 700;
	const int density = 25;
	const int prange = (100 / density);
	int* graph = (int*)calloc(sizeof(int), n * n);
	const int range = 1000;
	const int infinite = std::numeric_limits<int>::max();
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			// Set G[i][i] = 0.
			if (i == j) {
				graph[i * n + j] = 0;
				continue;
			}
			int pr = std::rand() % prange;
			// Set edge random edge weight to random value, or to infinite.
			graph[i * n + j] = pr == 0 ? ((rand() % range) + 1) : infinite;
		}
	}

	int* path = nullptr;
	
	int* graph_d;
	cudaMalloc(&graph_d, n * n);
	cudaMemcpy(graph_d, graph, n * n, cudaMemcpyHostToDevice);

	int* path_d;
	cudaMalloc(&path_d, n * n);

	const int grid = 1;
	const int block = 1;

	ParallelFloydWarshall <<<grid, block>>> (graph_d, n, path_d);

	cudaMemcpy(path, path_d, n * n, cudaMemcpyDeviceToHost);

	std::cout << path;

	// Free memory.
	cudaFree(path_d);
	cudaFree(graph_d);
	free(path);
	free(graph);
	return EXIT_SUCCESS;
}
