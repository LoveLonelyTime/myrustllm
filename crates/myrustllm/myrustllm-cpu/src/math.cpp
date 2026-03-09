#include <myrustllm-cpu.h>
#include "../include/math.h"
#include <immintrin.h>
#include <omp.h>
#include <tensor.h>

void cpu_tensor_copy_f32(CPUTensor lhs, CPUTensor rhs)
{
    size_t n_elements = tensor_numel(lhs);
#pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        int num_threads = omp_get_num_threads();
        int chunk_size = n_elements / num_threads;
        int start = thread_id * chunk_size;
        int end = (thread_id == num_threads - 1) ? n_elements : start + chunk_size;

        std::vector<size_t> lhs_idx = tensor_linear2idx(lhs, start);
        std::vector<size_t> rhs_idx = tensor_linear2idx(rhs, start);

        for (int i = start; i < end; i++)
        {
            size_t lhs_offset = tensor_idx2offset(lhs, lhs_idx);
            size_t rhs_offset = tensor_idx2offset(rhs, rhs_idx);
            TENSOR_PTR(lhs.data, float, lhs_offset) = TENSOR_PTR(rhs.data, float, rhs_offset);
            tensor_next_idx(lhs, lhs_idx);
            tensor_next_idx(rhs, rhs_idx);
        }
    }
}

void cpu_tensor_add_f32(CPUTensor out, CPUTensor lhs, CPUTensor rhs)
{
    size_t n_elements = tensor_numel(lhs);
#pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        int num_threads = omp_get_num_threads();
        int chunk_size = n_elements / num_threads;
        int start = thread_id * chunk_size;
        int end = (thread_id == num_threads - 1) ? n_elements : start + chunk_size;

        std::vector<size_t> lhs_idx = tensor_linear2idx(lhs, start);
        std::vector<size_t> rhs_idx = tensor_linear2idx(rhs, start);
        std::vector<size_t> out_idx = tensor_linear2idx(out, start);

        for (int i = start; i < end; i++)
        {
            size_t lhs_offset = tensor_idx2offset(lhs, lhs_idx);
            size_t rhs_offset = tensor_idx2offset(rhs, rhs_idx);
            size_t out_offset = tensor_idx2offset(out, out_idx);
            TENSOR_PTR(out.data, float, out_offset) = TENSOR_PTR(lhs.data, float, lhs_offset) + TENSOR_PTR(rhs.data, float, rhs_offset);
            tensor_next_idx(lhs, lhs_idx);
            tensor_next_idx(rhs, rhs_idx);
            tensor_next_idx(out, out_idx);
        }
    }
}

void cpu_tensor_add_f32_(CPUTensor lhs, CPUTensor rhs)
{
    size_t n_elements = tensor_numel(lhs);
#pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        int num_threads = omp_get_num_threads();
        int chunk_size = n_elements / num_threads;
        int start = thread_id * chunk_size;
        int end = (thread_id == num_threads - 1) ? n_elements : start + chunk_size;

        std::vector<size_t> lhs_idx = tensor_linear2idx(lhs, start);
        std::vector<size_t> rhs_idx = tensor_linear2idx(rhs, start);

        for (int i = start; i < end; i++)
        {
            size_t lhs_offset = tensor_idx2offset(lhs, lhs_idx);
            size_t rhs_offset = tensor_idx2offset(rhs, rhs_idx);
            TENSOR_PTR(lhs.data, float, lhs_offset) += TENSOR_PTR(rhs.data, float, rhs_offset);
            tensor_next_idx(lhs, lhs_idx);
            tensor_next_idx(rhs, rhs_idx);
        }
    }
}

void cpu_tensor_add_f32_r(CPUTensor lhs, CPUTensor rhs, size_t batch_dims)
{
    std::vector<size_t> batch_shape;
    std::vector<size_t> batch_stride;
    std::vector<size_t> reduce_shape;
    std::vector<size_t> reduce_stride;

    for (size_t dim = 0; dim < rhs.dims; dim++)
    {
        if (dim < batch_dims)
        {
            batch_shape.push_back(rhs.shape[dim]);
            batch_stride.push_back(rhs.stride[dim]);
        }
        else
        {
            reduce_shape.push_back(rhs.shape[dim]);
            reduce_stride.push_back(rhs.stride[dim]);
        }
    }
    CPUTensor batch_tensor = {nullptr, batch_shape.data(), batch_stride.data(), batch_dims};
    CPUTensor reduce_tensor = {nullptr, reduce_shape.data(), reduce_stride.data(), rhs.dims - batch_dims};
    size_t batch_size = tensor_numel(batch_tensor);
    size_t reduce_size = tensor_numel(reduce_tensor);

#pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        int num_threads = omp_get_num_threads();
        int chunk_size = batch_size / num_threads;
        int start = thread_id * chunk_size;
        int end = (thread_id == num_threads - 1) ? batch_size : start + chunk_size;

        std::vector<size_t> batch_idx = tensor_linear2idx(batch_tensor, start);
        std::vector<size_t> lhs_idx = tensor_linear2idx(lhs, start);

        for (int i = start; i < end; i++)
        {
            size_t batch_offset = tensor_idx2offset(batch_tensor, batch_idx);
            size_t lhs_offset = tensor_idx2offset(lhs, lhs_idx);

            float acc = 0;

#pragma omp parallel
            {
                int thread_id = omp_get_thread_num();
                int num_threads = omp_get_num_threads();
                int chunk_size = reduce_size / num_threads;
                int start = thread_id * chunk_size;
                int end = (thread_id == num_threads - 1) ? reduce_size : start + chunk_size;
                std::vector<size_t> reduce_idx = tensor_linear2idx(reduce_tensor, start);
                float acc_inner = 0;
                for (int i = start; i < end; i++)
                {
                    size_t reduce_offset = tensor_idx2offset(reduce_tensor, reduce_idx);
                    acc_inner += TENSOR_PTR(rhs.data, float, batch_offset + reduce_offset);
                    tensor_next_idx(reduce_tensor, reduce_idx);
                }

#pragma omp critical
                acc += acc_inner;
            }

            TENSOR_PTR(lhs.data, float, lhs_offset) = acc;

            tensor_next_idx(batch_tensor, batch_idx);
            tensor_next_idx(lhs, lhs_idx);
        }
    }
}

void cpu_tensor_matmul_f32(CPUTensor out, CPUTensor lhs, CPUTensor rhs)
{
    size_t NUM_DIM = out.dims;
    size_t M = out.shape[NUM_DIM - 2];
    size_t N = out.shape[NUM_DIM - 1];
    size_t K = lhs.shape[NUM_DIM - 1];

    size_t OUT_M_STRIDE = out.stride[NUM_DIM - 2];
    size_t OUT_N_STRIDE = out.stride[NUM_DIM - 1];
    size_t LHS_M_STRIDE = lhs.stride[NUM_DIM - 2];
    size_t LHS_K_STRIDE = lhs.stride[NUM_DIM - 1];
    size_t RHS_K_STRIDE = rhs.stride[NUM_DIM - 2];
    size_t RHS_N_STRIDE = rhs.stride[NUM_DIM - 1];

    std::vector<size_t> out_batch_shape;
    std::vector<size_t> out_batch_stride;
    std::vector<size_t> lhs_batch_shape;
    std::vector<size_t> lhs_batch_stride;
    std::vector<size_t> rhs_batch_shape;
    std::vector<size_t> rhs_batch_stride;

    for (size_t dim = 0; dim < NUM_DIM - 2; dim++)
    {
        out_batch_shape.push_back(out.shape[dim]);
        out_batch_stride.push_back(out.stride[dim]);
        lhs_batch_shape.push_back(lhs.shape[dim]);
        lhs_batch_stride.push_back(lhs.stride[dim]);
        rhs_batch_shape.push_back(rhs.shape[dim]);
        rhs_batch_stride.push_back(rhs.stride[dim]);
    }

    CPUTensor out_batch_tensor = {nullptr, out_batch_shape.data(), out_batch_stride.data(), NUM_DIM - 2};
    CPUTensor lhs_batch_tensor = {nullptr, lhs_batch_shape.data(), lhs_batch_stride.data(), NUM_DIM - 2};
    CPUTensor rhs_batch_tensor = {nullptr, rhs_batch_shape.data(), rhs_batch_stride.data(), NUM_DIM - 2};

    size_t batch_size = tensor_numel(out_batch_tensor);

#pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        int num_threads = omp_get_num_threads();
        int chunk_size = batch_size / num_threads;
        int start = thread_id * chunk_size;
        int end = (thread_id == num_threads - 1) ? batch_size : start + chunk_size;

        std::vector<size_t> out_batch_idx = tensor_linear2idx(out_batch_tensor, start);
        std::vector<size_t> lhs_batch_idx = tensor_linear2idx(lhs_batch_tensor, start);
        std::vector<size_t> rhs_batch_idx = tensor_linear2idx(rhs_batch_tensor, start);

        for (int i = start; i < end; i++)
        {
            size_t out_batch_offset = tensor_idx2offset(out_batch_tensor, out_batch_idx);
            size_t lhs_batch_offset = tensor_idx2offset(lhs_batch_tensor, lhs_batch_idx);
            size_t rhs_batch_offset = tensor_idx2offset(rhs_batch_tensor, rhs_batch_idx);

            float *OUT = &TENSOR_PTR(out.data, float, out_batch_offset);
            float *LHS = &TENSOR_PTR(lhs.data, float, lhs_batch_offset);
            float *RHS = &TENSOR_PTR(rhs.data, float, rhs_batch_offset);

#define BLOCK_SIZE 64
#pragma omp parallel for collapse(2)
            for (int ii = 0; ii < M; ii += BLOCK_SIZE)
            {
                for (int jj = 0; jj < N; jj += BLOCK_SIZE)
                {
                    for (int kk = 0; kk < K; kk += BLOCK_SIZE)
                    {
                        for (int i = ii; i < ((ii + BLOCK_SIZE) > M ? M : (ii + BLOCK_SIZE)); i++)
                        {
                            for (int k = kk; k < ((kk + BLOCK_SIZE) > K ? K : (kk + BLOCK_SIZE)); k++)
                            {
                                float r = LHS[i * LHS_M_STRIDE + k * LHS_K_STRIDE];
                                for (int j = jj; j < ((jj + BLOCK_SIZE) > N ? N : (jj + BLOCK_SIZE)); j++)
                                {
                                    OUT[i * OUT_M_STRIDE + j * OUT_N_STRIDE] += r * RHS[k * RHS_K_STRIDE + j * RHS_N_STRIDE];
                                }
                            }
                        }
                    }
                }
            }

            tensor_next_idx(out_batch_tensor, out_batch_idx);
            tensor_next_idx(lhs_batch_tensor, lhs_batch_idx);
            tensor_next_idx(rhs_batch_tensor, rhs_batch_idx);
        }
    }
}