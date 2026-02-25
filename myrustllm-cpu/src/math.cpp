#include <myrustllm-cpu.h>
#include "../include/math.h"

void cpu_tensor_add_f32(
    const float* a, const float* b, float* out,
    const size_t* a_strides, const size_t* b_strides, const size_t* out_strides,
    const size_t* shape,
    size_t n_dims,
    size_t n_elements
){
    #pragma omp parallel for
    for(size_t linear_idx = 0; linear_idx < n_elements; linear_idx++) {
        size_t a_offset = linear2tensor(linear_idx, a_strides, shape, n_dims);
        size_t b_offset = linear2tensor(linear_idx, b_strides, shape, n_dims);
        size_t out_offset = linear2tensor(linear_idx, out_strides, shape, n_dims);

        out[out_offset] = a[a_offset] + b[b_offset];
    }
}

void cpu_tensor_add_f32_(
    float* a, const float* b,
    const size_t* a_strides, const size_t* b_strides,
    const size_t* shape,
    size_t n_dims,
    size_t n_elements
){
    #pragma omp parallel for
    for(size_t linear_idx = 0; linear_idx < n_elements; linear_idx++) {
        size_t a_offset = linear2tensor(linear_idx, a_strides, shape, n_dims);
        size_t b_offset = linear2tensor(linear_idx, b_strides, shape, n_dims);

        a[a_offset] += b[b_offset];
    }
}