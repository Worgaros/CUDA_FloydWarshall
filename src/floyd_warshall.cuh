#pragma once

#include <cuda.h>
#include <cuda_runtime.h>/

__global__ void parallel_floyd_warshall(int* graph, int n, int* path);