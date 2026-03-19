#ifndef __BINARY_OPS_H__
#define __BINARY_OPS_H__

#include <omp.h>
#include <stdexcept>
#include "myrustllm-cpu.h"
#include "tensor.h"

template <typename T, typename Op>
void tensor_binary_op_impl(CPUTensor& out, CPUTensor& lhs, CPUTensor& rhs,
                           Op op) {
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

    for (int i = start; i < end; i++) {
      size_t lhs_offset = tensor_idx2offset(lhs, lhs_idx);
      size_t rhs_offset = tensor_idx2offset(rhs, rhs_idx);
      size_t out_offset = tensor_idx2offset(out, out_idx);

      T lhs_data = tensor_data<T>(lhs)[lhs_offset];
      T rhs_data = tensor_data<T>(rhs)[rhs_offset];
      T* out_data = &tensor_data<T>(out)[out_offset];

      *out_data = op(lhs_data, rhs_data);

      tensor_next_idx(lhs, lhs_idx);
      tensor_next_idx(rhs, rhs_idx);
      tensor_next_idx(out, out_idx);
    }
  }
}

template <typename Op>
void tensor_binary_op_dispatch(CPUTensor& out, CPUTensor& lhs, CPUTensor& rhs,
                               Op&& op, DType dtype) {
  switch (dtype) {
    case DT_F32:
      tensor_binary_op_impl<float>(out, lhs, rhs, std::forward<Op>(op));
      break;
    case DT_F64:
      tensor_binary_op_impl<double>(out, lhs, rhs, std::forward<Op>(op));
      break;
    case DT_I32:
      tensor_binary_op_impl<int32_t>(out, lhs, rhs, std::forward<Op>(op));
      break;
    case DT_I64:
      tensor_binary_op_impl<int64_t>(out, lhs, rhs, std::forward<Op>(op));
      break;
    default:
      throw std::runtime_error("Unsupported data type.");
  }
}

#endif  // __BINARY_OPS_H__
