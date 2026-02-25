#ifndef GEMM_H
#define GEMM_H

#include <cuda_runtime.h>
#include <math.cuh>

template <size_t M_BLOCK_SIZE, size_t N_BLOCK_SIZE, size_t K_BLOCK_SIZE, size_t TM_BLOCK_SIZE, size_t TN_BLOCK_SIZE>
__global__ void gemm_f32_kernel(
    const float *a,
    const float *b,
    float *out,
    const size_t *a_strides, const size_t *b_strides, const size_t *out_strides,
    const size_t *a_shape,
    const size_t *b_shape,
    size_t n_batchs,
    size_t n_dims)
{
    size_t m_len = a_shape[n_dims - 2];
    size_t n_len = b_shape[n_dims - 1];
    size_t k_len = a_shape[n_dims - 1];
    size_t a_m_stride = a_strides[n_dims - 2];
    size_t a_k_stride = a_strides[n_dims - 1];
    size_t b_k_stride = a_strides[n_dims - 2];
    size_t b_n_stride = a_strides[n_dims - 1];
    size_t out_m_stide = out_strides[n_dims - 2];
    size_t out_n_stide = out_strides[n_dims - 1];

    // Calc offsets of batch
    size_t linear_idx = blockIdx.z;
    for (; linear_idx < n_batchs; linear_idx += gridDim.z)
    {
        size_t a_offset = linear2tensor(linear_idx, a_strides, a_shape, n_dims - 2);
        size_t b_offset = linear2tensor(linear_idx, b_strides, a_shape, n_dims - 2);
        size_t out_offset = linear2tensor(linear_idx, out_strides, a_shape, n_dims - 2);

        size_t m_start = M_BLOCK_SIZE * blockIdx.y;
        size_t n_start = N_BLOCK_SIZE * blockIdx.x;
        size_t pid = threadIdx.x + blockDim.x * threadIdx.y;
        size_t num_k_blocks = (k_len + K_BLOCK_SIZE - 1) / K_BLOCK_SIZE;

        __shared__ float s_a[M_BLOCK_SIZE][K_BLOCK_SIZE];
        __shared__ float s_b[K_BLOCK_SIZE][N_BLOCK_SIZE];
        float s_out[TM_BLOCK_SIZE][TN_BLOCK_SIZE] = {0.0f};

        size_t tm_start = pid / (N_BLOCK_SIZE / TN_BLOCK_SIZE) * TM_BLOCK_SIZE;
        size_t tn_start = pid % (N_BLOCK_SIZE / TN_BLOCK_SIZE) * TN_BLOCK_SIZE;

        size_t a_m_offset = pid / K_BLOCK_SIZE;
        size_t a_k_offset = pid % K_BLOCK_SIZE;

        size_t b_k_offset = pid / N_BLOCK_SIZE;
        size_t b_n_offset = pid % N_BLOCK_SIZE;

        // Iter k-block
        for (size_t kk = 0; kk < num_k_blocks; kk++)
        {
            size_t k_start = kk * K_BLOCK_SIZE;

            // Load a and b
            if (m_start + a_m_offset < m_len && k_start + a_k_offset < k_len)
            {
                s_a[a_m_offset][a_k_offset] = a[a_offset + (m_start + a_m_offset) * a_m_stride + (k_start + a_k_offset) * a_k_stride];
            }
            else
            {
                s_a[a_m_offset][a_k_offset] = 0.0f;
            }

            if (n_start + b_n_offset < n_len && k_start + b_k_offset < k_len)
            {
                s_b[b_k_offset][b_n_offset] = b[b_offset + (k_start + b_k_offset) * b_k_stride + (n_start + b_n_offset) * b_n_stride];
            }
            else
            {
                s_b[b_k_offset][b_n_offset] = 0.0f;
            }

            __syncthreads();

            // Dot product
            for (size_t k = 0; k < K_BLOCK_SIZE; k++)
            {
                for (size_t tm_offset = 0; tm_offset < TM_BLOCK_SIZE; tm_offset++)
                {
                    for (size_t tn_offset = 0; tn_offset < TN_BLOCK_SIZE; tn_offset++)
                    {
                        s_out[tm_offset][tn_offset] += s_a[tm_start + tm_offset][k] * s_b[k][tn_start + tn_offset];
                    }
                }
            }

            __syncthreads();
        }

        // Store out
        for (size_t tm_offset = 0; tm_offset < TM_BLOCK_SIZE; tm_offset++)
        {
            for (size_t tn_offset = 0; tn_offset < TN_BLOCK_SIZE; tn_offset++)
            {
                if (m_start + tm_start + tm_offset < m_len && n_start + tn_start + tn_offset < n_len)
                {
                    out[out_offset + (m_start + tm_start + tm_offset) * out_m_stide + (n_start + tn_start + tn_offset) * out_n_stide] = s_out[tm_offset][tn_offset];
                }
            }
        }
    }
}

#endif