#ifndef __REDUCE_OPS_H__
#define __REDUCE_OPS_H__

#include <omp.h>
#include <stdexcept>
#include <vector>
#include "myrustllm-cpu.h"
#include "tensor.h"

template <typename T, typename Op>
void tensor_reduce_op_impl(CPUTensor& lhs, CPUTensor& rhs, size_t batch_dims) {
  std::vector<size_t> batch_shape;
  std::vector<size_t> batch_stride;
  std::vector<size_t> reduce_shape;
  std::vector<size_t> reduce_stride;

  for (size_t dim = 0; dim < rhs.dims; dim++) {
    if (dim < batch_dims) {
      batch_shape.push_back(rhs.shape[dim]);
      batch_stride.push_back(rhs.stride[dim]);
    } else {
      reduce_shape.push_back(rhs.shape[dim]);
      reduce_stride.push_back(rhs.stride[dim]);
    }
  }
  CPUTensor batch_tensor = {nullptr, batch_shape.data(), batch_stride.data(),
                            batch_dims};
  CPUTensor reduce_tensor = {nullptr, reduce_shape.data(), reduce_stride.data(),
                             rhs.dims - batch_dims};
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

    for (int i = start; i < end; i++) {
      size_t batch_offset = tensor_idx2offset(batch_tensor, batch_idx);
      size_t lhs_offset = tensor_idx2offset(lhs, lhs_idx);

      T acc = Op::template init<T>();

#pragma omp parallel
      {
        int thread_id = omp_get_thread_num();
        int num_threads = omp_get_num_threads();
        int chunk_size = reduce_size / num_threads;
        int start = thread_id * chunk_size;
        int end =
            (thread_id == num_threads - 1) ? reduce_size : start + chunk_size;
        std::vector<size_t> reduce_idx =
            tensor_linear2idx(reduce_tensor, start);
        T acc_inner = Op::template init<T>();
        for (int i = start; i < end; i++) {
          size_t reduce_offset = tensor_idx2offset(reduce_tensor, reduce_idx);
          acc =
              Op::template apply<T>(acc, tensor_data<T>(rhs)[batch_offset + reduce_offset]);
          tensor_next_idx(reduce_tensor, reduce_idx);
        }

#pragma omp critical
        acc = Op::template apply<T>(acc, acc_inner);
      }

      tensor_data<T>(lhs)[lhs_offset] = acc;

      tensor_next_idx(batch_tensor, batch_idx);
      tensor_next_idx(lhs, lhs_idx);
    }
  }
}

template <typename Op>
void tensor_reduce_op_dispatch(CPUTensor& lhs, CPUTensor& rhs,
                               size_t batch_dims) {
  switch (lhs.dtype) {
    case DT_F32:
      tensor_reduce_op_impl<float, Op>(lhs, rhs, batch_dims);
      break;
    case DT_F64:
      tensor_reduce_op_impl<double, Op>(lhs, rhs, batch_dims);
      break;
    case DT_I32:
      tensor_reduce_op_impl<int32_t, Op>(lhs, rhs, batch_dims);
      break;
    case DT_I64:
      tensor_reduce_op_impl<int64_t, Op>(lhs, rhs, batch_dims);
      break;
    default:
      throw std::runtime_error("Unsupported data type.");
  }
}

#endif