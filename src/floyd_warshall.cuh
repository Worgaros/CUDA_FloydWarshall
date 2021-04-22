#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

__global__ void floyd_warshall_buffer(int* in_mat, int* in_mat_t,
const int* in_x, const int* in_y);
