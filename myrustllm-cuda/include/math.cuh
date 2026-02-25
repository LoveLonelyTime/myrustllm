#ifndef ELEMENTWISE_H
#define ELEMENTWISE_H

#include <cuda_runtime.h>

__forceinline__ __device__ size_t linear2tensor(
    size_t linear_idx,
    const size_t *strides,
    const size_t *shape,
    size_t n_dims)
{
    size_t offset = 0;

    for (size_t dim_r = 0; dim_r < n_dims; dim_r++)
    {
        size_t dim = n_dims - dim_r - 1;
        size_t dim_size = shape[dim];
        size_t dim_idx = linear_idx % dim_size;

        linear_idx /= dim_size;

        offset += dim_idx * strides[dim];
    }

    return offset;
}

template <typename T>
__global__ void elementwise_copy_kernel(
    const T *a,
    T *out,
    const size_t *a_strides,
    const size_t *out_strides,
    const size_t *shape,
    size_t n_dims,
    size_t n_elements)
{
    size_t thread_stride = gridDim.x * blockDim.x;
    size_t linear_idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (; linear_idx < n_elements; linear_idx += thread_stride)
    {
        size_t a_offset = linear2tensor(linear_idx, a_strides, shape, n_dims);
        size_t out_offset = linear2tensor(linear_idx, out_strides, shape, n_dims);
        out[out_offset] = a[a_offset];
    }
}

template <typename T>
__global__ void elementwise_add_kernel(
    const T *a, const T *b, T *out,
    const size_t *a_strides, const size_t *b_strides, const size_t *out_strides,
    const size_t *shape,
    size_t n_dims,
    size_t n_elements)
{
    size_t thread_stride = gridDim.x * blockDim.x;
    size_t linear_idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (; linear_idx < n_elements; linear_idx += thread_stride)
    {
        size_t a_offset = linear2tensor(linear_idx, a_strides, shape, n_dims);
        size_t b_offset = linear2tensor(linear_idx, b_strides, shape, n_dims);
        size_t out_offset = linear2tensor(linear_idx, out_strides, shape, n_dims);

        out[out_offset] = a[a_offset] + b[b_offset];
    }
}

template <typename T>
__global__ void elementwise_sub_kernel(
    const T *a, const T *b, T *out,
    const size_t *a_strides, const size_t *b_strides, const size_t *out_strides,
    const size_t *shape,
    size_t n_dims,
    size_t n_elements)
{
    size_t thread_stride = gridDim.x * blockDim.x;
    size_t linear_idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (; linear_idx < n_elements; linear_idx += thread_stride)
    {
        size_t a_offset = linear2tensor(linear_idx, a_strides, shape, n_dims);
        size_t b_offset = linear2tensor(linear_idx, b_strides, shape, n_dims);
        size_t out_offset = linear2tensor(linear_idx, out_strides, shape, n_dims);

        out[out_offset] = a[a_offset] - b[b_offset];
    }
}

template <typename T>
__global__ void elementwise_mul_kernel(
    const T *a, const T *b, T *out,
    const size_t *a_strides, const size_t *b_strides, const size_t *out_strides,
    const size_t *shape,
    size_t n_dims,
    size_t n_elements)
{
    size_t thread_stride = gridDim.x * blockDim.x;
    size_t linear_idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (; linear_idx < n_elements; linear_idx += thread_stride)
    {
        size_t a_offset = linear2tensor(linear_idx, a_strides, shape, n_dims);
        size_t b_offset = linear2tensor(linear_idx, b_strides, shape, n_dims);
        size_t out_offset = linear2tensor(linear_idx, out_strides, shape, n_dims);

        out[out_offset] = a[a_offset] * b[b_offset];
    }
}

template <typename T>
__global__ void elementwise_div_kernel(
    const T *a, const T *b, T *out,
    const size_t *a_strides, const size_t *b_strides, const size_t *out_strides,
    const size_t *shape,
    size_t n_dims,
    size_t n_elements)
{
    size_t thread_stride = gridDim.x * blockDim.x;
    size_t linear_idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (; linear_idx < n_elements; linear_idx += thread_stride)
    {
        size_t a_offset = linear2tensor(linear_idx, a_strides, shape, n_dims);
        size_t b_offset = linear2tensor(linear_idx, b_strides, shape, n_dims);
        size_t out_offset = linear2tensor(linear_idx, out_strides, shape, n_dims);

        out[out_offset] = a[a_offset] / b[b_offset];
    }
}

template <typename T>
__global__ void elementwise_neg_kernel(
    const T *a,
    T *out,
    const size_t *a_strides,
    const size_t *out_strides,
    const size_t *shape,
    size_t n_dims,
    size_t n_elements)
{
    size_t thread_stride = gridDim.x * blockDim.x;
    size_t linear_idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (; linear_idx < n_elements; linear_idx += thread_stride)
    {
        size_t a_offset = linear2tensor(linear_idx, a_strides, shape, n_dims);
        size_t out_offset = linear2tensor(linear_idx, out_strides, shape, n_dims);
        out[out_offset] = -a[a_offset];
    }
}

#endif // ELEMENTWISE_H
