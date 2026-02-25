#ifndef MYRUSTLLM_CPU_H
#define MYRUSTLLM_CPU_H

#include <stdlib.h>

#if defined(_WIN32)
#define __MYRUSTLLM_API__ __declspec(dlllimport)
#elif defined(__GNUC__)
#define __MYRUSTLLM_API__ __attribute__((visibility("default")))
#else
#define __MYRUSTLLM_API__
#endif

#ifdef __cplusplus
extern "C"
{
#endif

    // ---------- Math ----------
    // FP32 ADD
    __MYRUSTLLM_API__ void cpu_tensor_add_f32(
        const float *a, const float *b, float *out,
        const size_t *a_strides, const size_t *b_strides, const size_t *out_strides,
        const size_t *shape,
        size_t n_dims,
        size_t n_elements);

    __MYRUSTLLM_API__ void cpu_tensor_add_f32_(
        float* a, const float* b,
        const size_t* a_strides, const size_t* b_strides,
        const size_t* shape,
        size_t n_dims,
        size_t n_elements
    );

#ifdef __cplusplus
}
#endif

#endif // MYRUSTLLM_CPU_H