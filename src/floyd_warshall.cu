#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <memory>

#include "floyd_warshall.cuh"

__global__ void floyd_warshall_buffer(int* in_mat, int* in_mat_t,
const int* in_x, const int* in_y) {
	const int dx = threadIdx.x;
	const int dy = threadIdx.y;
	const int mx = gridDim.x;
	const int my = gridDim.y;

	int position = dy * mx + dx;
	int position_t = dx * my + dy;
	float val1 = in_mat[position];
	float val2 = in_x[dx] + in_y[dy];
	in_mat[position] = (val1 < val2) ? val1 : val2;
	in_mat_t[position_t] = (val1 < val2) ? val1 : val2;
}

int main() {
	const int n = 3;
	const int infinite = std::numeric_limits<int>::max();

	int* graph = (int*)malloc(n * n * sizeof(int));
	graph[0] = 0;
	graph[1] = infinite;
	graph[2] = 1;
	graph[3] = 12;
	graph[4] = 0;
	graph[5] = 12;
	graph[6] = 1;
	graph[7] = 25;
	graph[8] = 0;

	int* graph_t = (int*)malloc(n * n * sizeof(int));
	graph_t[0] = 0;
	graph_t[1] = 12;
	graph_t[2] = 1;
	graph_t[3] = infinite;
	graph_t[4] = 0;
	graph_t[5] = 25;
	graph_t[6] = 1;
	graph_t[7] = 12;
	graph_t[8] = 0;

	int* path = (int*)calloc(n * n, sizeof(int));

	int* graph_d;
	int* graph_t_d;
	int* in_x_d;
	int* in_y_d;
	cudaMalloc(&graph_d, n * n * sizeof(int));
	cudaMalloc(&graph_t_d, n * n * sizeof(int));
	cudaMalloc(&in_x_d, n * sizeof(int));
	cudaMalloc(&in_y_d, n * sizeof(int));
	cudaMemcpy(graph_d, graph, n * n * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(graph_t_d, graph_t, n * n * sizeof(int), cudaMemcpyHostToDevice);

	const dim3 grid = {n, n, 1};
	const int block = 9;

	for (int i = 0; i < n; ++i) {
		// Copy in in_x_d the i line from graph_d. 
		cudaMemcpy(in_x_d, &graph_d[i * n], n * sizeof(int), cudaMemcpyDeviceToDevice);
		// Copy in in_y_d the i line from graph_t_d. 
		cudaMemcpy(in_y_d, &graph_t_d[i * n], n * sizeof(int), cudaMemcpyDeviceToDevice);
		floyd_warshall_buffer<<<grid, block>>>(graph_d, graph_t_d, in_x_d, in_y_d);
	}

	cudaMemcpy(path, graph_d, n * n * sizeof(int), cudaMemcpyDeviceToHost);

	for (int x = 0; x < n; ++x)
	{
		for (int y = 0; y < n; ++y)
		{
				std::cout << path[x * n + y];
				std::cout << " ";
		}
		std::cout << "\n";
	}

	// Free memory.
	cudaFree(graph_t_d);
	cudaFree(graph_d);
	cudaFree(in_x_d);
	cudaFree(in_y_d);
	free(graph);
	free(graph_t);
	return EXIT_SUCCESS;
}
