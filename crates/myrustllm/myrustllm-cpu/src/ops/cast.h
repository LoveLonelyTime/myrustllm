#ifndef __CAST_H__
#define __CAST_H__

#include "tensor.h"
#include <omp.h>

template <typename O, typename L>
void tensor_cast_impl(CPUTensor &out, CPUTensor &lhs)
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
        std::vector<size_t> out_idx = tensor_linear2idx(out, start);

        for (int i = start; i < end; i++)
        {
            size_t lhs_offset = tensor_idx2offset(lhs, lhs_idx);
            size_t out_offset = tensor_idx2offset(out, out_idx);

            L lhs_data = tensor_data<L>(lhs)[lhs_offset];
            O *out_data = &tensor_data<O>(out)[out_offset];

            *out_data = static_cast<O>(lhs_data);

            tensor_next_idx(lhs, lhs_idx);
            tensor_next_idx(out, out_idx);
        }
    }
}

template <DType O, DType L>
void tensor_cast_dispatch(CPUTensor &out, CPUTensor &lhs)
{
    using TO = typename dtype_to_type<O>::type;
    using TL = typename dtype_to_type<L>::type;
    tensor_cast_impl<TO, TL>(out, lhs);
}

template <typename T>
void tensor_copy_impl(CPUTensor &dst, CPUTensor &src)
{
    size_t n_elements = tensor_numel(dst);
#pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        int num_threads = omp_get_num_threads();
        int chunk_size = n_elements / num_threads;
        int start = thread_id * chunk_size;
        int end = (thread_id == num_threads - 1) ? n_elements : start + chunk_size;

        std::vector<size_t> dst_idx = tensor_linear2idx(dst, start);
        std::vector<size_t> src_idx = tensor_linear2idx(src, start);

        for (int i = start; i < end; i++)
        {
            size_t dst_offset = tensor_idx2offset(dst, dst_idx);
            size_t src_offset = tensor_idx2offset(src, src_idx);
            tensor_data<T>(dst)[dst_offset] = tensor_data<T>(src)[src_offset];
            tensor_next_idx(dst, dst_idx);
            tensor_next_idx(src, src_idx);
        }
    }
}

#endif
