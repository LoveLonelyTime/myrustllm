#ifndef MYRUSTLLM_CUDA_H
#define MYRUSTLLM_CUDA_H

#include <cuda_runtime.h>

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
    // ---------- Memory management ----------

    // Allocate tensor memory
    __MYRUSTLLM_API__ void *cuda_tensor_alloc(size_t count);
    // Free tensor memory
    __MYRUSTLLM_API__ void cuda_tensor_destroy(void *ptr);
    // Transfer data from the host
    __MYRUSTLLM_API__ void cuda_tensor_copy_from_host(void *dst, const void *src, size_t count);
    // Transfer data from the device
    __MYRUSTLLM_API__ void cuda_tensor_copy_from_device(void *dst, const void *src, size_t count);

    // ---------- Math ----------
    // FP32 COPY
    __MYRUSTLLM_API__ void cuda_tensor_copy_f32(
        const float *a, float *out,
        const size_t *a_strides, const size_t *out_strides,
        const size_t *shape,
        size_t n_dims,
        size_t n_elements);

    // FP32 ADD
    __MYRUSTLLM_API__ void cuda_tensor_add_f32(
        const float *a, const float *b, float *out,
        const size_t *a_strides, const size_t *b_strides, const size_t *out_strides,
        const size_t *shape,
        size_t n_dims,
        size_t n_elements);

    // FP32 SUB
    __MYRUSTLLM_API__ void cuda_tensor_sub_f32(
        const float *a, const float *b, float *out,
        const size_t *a_strides, const size_t *b_strides, const size_t *out_strides,
        const size_t *shape,
        size_t n_dims,
        size_t n_elements);

    // FP32 MUL
    __MYRUSTLLM_API__ void cuda_tensor_mul_f32(
        const float *a, const float *b, float *out,
        const size_t *a_strides, const size_t *b_strides, const size_t *out_strides,
        const size_t *shape,
        size_t n_dims,
        size_t n_elements);

    // FP32 SUB
    __MYRUSTLLM_API__ void cuda_tensor_div_f32(
        const float *a, const float *b, float *out,
        const size_t *a_strides, const size_t *b_strides, const size_t *out_strides,
        const size_t *shape,
        size_t n_dims,
        size_t n_elements);

    // FP32 NEG
    __MYRUSTLLM_API__ void cuda_tensor_neg_f32(
        const float *a, float *out,
        const size_t *a_strides, const size_t *out_strides,
        const size_t *shape,
        size_t n_dims,
        size_t n_elements);

    // FP32 GEMM
    __MYRUSTLLM_API__ void cuda_tensor_gemm_f32(
        const float *a,
        const float *b,
        float *out,
        const size_t *a_strides, const size_t *b_strides, const size_t *out_strides,
        const size_t *a_shape,
        const size_t *b_shape,
        size_t n_batchs,
        size_t n_dims);

#ifdef __cplusplus
}
#endif

#endif // MYRUSTLLM_CUDA_H