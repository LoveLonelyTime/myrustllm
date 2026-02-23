#include <myrustllm-cuda.h>
#include <check.h>
#include <math.cuh>

void cuda_tensor_add_f32(
    const float* a, const float* b, float* out,
    const size_t* a_strides, const size_t* b_strides, const size_t* out_strides,
    const size_t* shape,
    size_t n_dims,
    size_t n_elements
) {
    size_t* a_strides_cuda, *b_strides_cuda, *out_strides_cuda, *shape_cuda;
    CUDA_CHECK(cudaMalloc(&a_strides_cuda, sizeof(size_t) * n_dims));
    CUDA_CHECK(cudaMalloc(&b_strides_cuda, sizeof(size_t) * n_dims));
    CUDA_CHECK(cudaMalloc(&out_strides_cuda, sizeof(size_t) * n_dims));
    CUDA_CHECK(cudaMalloc(&shape_cuda, sizeof(size_t) * n_dims));

    CUDA_CHECK(cudaMemcpy(a_strides_cuda, a_strides, sizeof(size_t) * n_dims, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b_strides_cuda, b_strides, sizeof(size_t) * n_dims, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(out_strides_cuda, out_strides, sizeof(size_t) * n_dims, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(shape_cuda, shape, sizeof(size_t) * n_dims, cudaMemcpyHostToDevice));
    
    LAUNCH_KERNEL(elementwise_add_kernel<float>, 32, 512, a, b, out, a_strides_cuda, b_strides_cuda, out_strides_cuda, shape_cuda, n_dims, n_elements);
    
    CUDA_CHECK(cudaFree(a_strides_cuda));
    CUDA_CHECK(cudaFree(b_strides_cuda));
    CUDA_CHECK(cudaFree(out_strides_cuda));
    CUDA_CHECK(cudaFree(shape_cuda));
}