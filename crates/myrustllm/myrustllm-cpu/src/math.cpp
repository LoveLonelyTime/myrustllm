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