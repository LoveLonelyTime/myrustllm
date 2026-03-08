#include <myrustllm-cuda.h>
#include <check.h>
#include <gemm.cuh>

void cuda_tensor_gemm_f32(
    const float *a,
    const float *b,
    float *out,
    const size_t *a_strides, const size_t *b_strides, const size_t *out_strides,
    const size_t *a_shape,
    const size_t *b_shape,
    size_t n_batchs,
    size_t n_dims)
{
    size_t *a_strides_cuda, *b_strides_cuda, *out_strides_cuda, *a_shape_cuda, *b_shape_cuda;
    CUDA_CHECK(cudaMalloc(&a_strides_cuda, sizeof(size_t) * n_dims));
    CUDA_CHECK(cudaMalloc(&b_strides_cuda, sizeof(size_t) * n_dims));
    CUDA_CHECK(cudaMalloc(&out_strides_cuda, sizeof(size_t) * n_dims));
    CUDA_CHECK(cudaMalloc(&a_shape_cuda, sizeof(size_t) * n_dims));
    CUDA_CHECK(cudaMalloc(&b_shape_cuda, sizeof(size_t) * n_dims));

    CUDA_CHECK(cudaMemcpy(a_strides_cuda, a_strides, sizeof(size_t) * n_dims, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b_strides_cuda, b_strides, sizeof(size_t) * n_dims, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(out_strides_cuda, out_strides, sizeof(size_t) * n_dims, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(a_shape_cuda, a_shape, sizeof(size_t) * n_dims, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b_shape_cuda, b_shape, sizeof(size_t) * n_dims, cudaMemcpyHostToDevice));

    size_t M = a_shape[n_dims - 2];
    size_t N = b_shape[n_dims - 1];
    size_t K = a_shape[n_dims - 1];

    const size_t THREAD_PER_BLOCK_EDGE = 32; // 1024 = 32 * 32 Threads
    const size_t TM_BLOCK_SIZE = 4;
    const size_t TN_BLOCK_SIZE = 4;

    const size_t M_BLOCK_SIZE = THREAD_PER_BLOCK_EDGE * TM_BLOCK_SIZE;
    const size_t N_BLOCK_SIZE = THREAD_PER_BLOCK_EDGE * TN_BLOCK_SIZE;
    const size_t K_BLOCK_SIZE = THREAD_PER_BLOCK_EDGE / TM_BLOCK_SIZE;

    dim3 block_dim((N + N_BLOCK_SIZE - 1) / N_BLOCK_SIZE, (M + M_BLOCK_SIZE - 1) / M_BLOCK_SIZE);
    dim3 thread_dim(N_BLOCK_SIZE / TN_BLOCK_SIZE, M_BLOCK_SIZE / TM_BLOCK_SIZE);

    LAUNCH_KERNEL((gemm_f32_kernel<M_BLOCK_SIZE, N_BLOCK_SIZE, K_BLOCK_SIZE, TM_BLOCK_SIZE, TN_BLOCK_SIZE>), block_dim, thread_dim, a, b, out, a_strides, b_strides, out_strides, a_shape, b_shape, n_batchs, n_dims);

    CUDA_CHECK(cudaFree(a_strides_cuda));
    CUDA_CHECK(cudaFree(b_strides_cuda));
    CUDA_CHECK(cudaFree(out_strides_cuda));
    CUDA_CHECK(cudaFree(a_shape_cuda));
    CUDA_CHECK(cudaFree(b_shape_cuda));
}