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

#pragma omp parallel for
    for (size_t batch_i = 0; batch_i < batch_size; batch_i++)
    {

        // float acc = 0.0;
        // #pragma omp parallel for reduction(+:acc)
        // for(size_t reduce_i = 0; reduce_i < reduce_size; reduce_i++) {
        //     acc += a[batch_i * a_strides[0] + reduce_i * a_strides[1]];
        // }
        // out[batch_i * b_strides[0]] = avx_sum(&a[batch_i * a_strides[0]], reduce_size);
        // out[batch_i * b_strides[0]] = acc;
    }
}