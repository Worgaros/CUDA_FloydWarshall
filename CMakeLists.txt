cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(CUDA_FloydWarshall LANGUAGES CXX CUDA)

set (CMAKE_CXX_STANDARD 17)
set (CMAKE_CXX_STANDARD_REQUIRED ON)

file(GLOB_RECURSE CUDA_FloydWarshall_SRC src/*.cu src/*.cuh)

add_executable(Executable ${CUDA_FloydWarshall_SRC})