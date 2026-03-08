#ifndef CHECK_H
#define CHECK_H

#include <cstdlib>
#include <cstdio>
#include <cuda_runtime.h>


#define CUDA_CHECK(call)                                                                 \
    do {                                                                                 \
        cudaError_t err = call;                                                          \
        if (err != cudaSuccess) {                                                        \
            fprintf(stderr, "CUDA error at %s:%d in function %s: %s\n",                  \
                    __FILE__, __LINE__, __FUNCTION__, cudaGetErrorString(err));          \
            exit(EXIT_FAILURE);                                                          \
        }                                                                                \
    } while(0);                                                                          \

#define LAUNCH_KERNEL(kernel, grid, block, ...)                                          \
    do {                                                                                 \
        kernel<<<grid, block>>>(__VA_ARGS__);                                            \
        cudaError_t err = cudaGetLastError();                                            \
        if (err != cudaSuccess) {                                                        \
            fprintf(stderr, "CUDA Error in %s at %s:%d in function %s: %s\n",            \
                    #kernel, __FILE__, __LINE__, __FUNCTION__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                                          \
        }                                                                                \
    } while(0);                                                                          \


#endif // CHECK_H