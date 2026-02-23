#ifndef ELEMENTWISE_H
#define ELEMENTWISE_H

#include <cuda_runtime.h>

template<typename T>
__global__ void elementwise_add_kernel(
    const T* a, const T* b, T* out,
    const size_t* a_strides, const size_t* b_strides, const size_t* out_strides,
    const size_t* shape,
    size_t n_dims,
    size_t n_elements
) {
    size_t thread_stride = gridDim.x * blockDim.x;
    size_t linear_idx = blockIdx.x * blockDim.x + threadIdx.x;
    for(;linear_idx < n_elements; linear_idx += thread_stride) {
        size_t a_offset = 0, b_offset = 0, out_offset = 0;
        size_t remaining = linear_idx;

        for(size_t dim_r = 0; dim_r < n_dims; dim_r++) {
            size_t dim = n_dims - dim_r - 1;
            size_t dim_size = shape[dim];
            size_t dim_idx = remaining % dim_size;

            remaining /= dim_size;

            a_offset += dim_idx * a_strides[dim];
            b_offset += dim_idx * b_strides[dim];
            out_offset += dim_idx * out_strides[dim];
        }

        out[out_offset] = a[a_offset] + b[b_offset];
    }
}

// template<typename T>
// __global__ void elementwise_neg_kernel(
//     const T* a,
//     T* out,
//     const size_t* a_strides,
//     onst size_t* out_strides,



#endif // ELEMENTWISE_H
