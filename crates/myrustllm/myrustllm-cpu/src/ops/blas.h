#ifndef __BLAS_H__
#define __BLAS_H__

#include <omp.h>
#include <stdexcept>
#include "myrustllm-cpu.h"
#include "tensor.h"

template <typename T>
void tensor_matmul_impl(CPUTensor& out, CPUTensor& lhs, CPUTensor& rhs) {
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

  for (size_t dim = 0; dim < NUM_DIM - 2; dim++) {
    out_batch_shape.push_back(out.shape[dim]);
    out_batch_stride.push_back(out.stride[dim]);
    lhs_batch_shape.push_back(lhs.shape[dim]);
    lhs_batch_stride.push_back(lhs.stride[dim]);
    rhs_batch_shape.push_back(rhs.shape[dim]);
    rhs_batch_stride.push_back(rhs.stride[dim]);
  }

  CPUTensor out_batch_tensor = {nullptr, out_batch_shape.data(),
                                out_batch_stride.data(), NUM_DIM - 2};
  CPUTensor lhs_batch_tensor = {nullptr, lhs_batch_shape.data(),
                                lhs_batch_stride.data(), NUM_DIM - 2};
  CPUTensor rhs_batch_tensor = {nullptr, rhs_batch_shape.data(),
                                rhs_batch_stride.data(), NUM_DIM - 2};

  size_t batch_size = tensor_numel(out_batch_tensor);

#pragma omp parallel
  {
    int thread_id = omp_get_thread_num();
    int num_threads = omp_get_num_threads();
    int chunk_size = batch_size / num_threads;
    int start = thread_id * chunk_size;
    int end = (thread_id == num_threads - 1) ? batch_size : start + chunk_size;

    std::vector<size_t> out_batch_idx =
        tensor_linear2idx(out_batch_tensor, start);
    std::vector<size_t> lhs_batch_idx =
        tensor_linear2idx(lhs_batch_tensor, start);
    std::vector<size_t> rhs_batch_idx =
        tensor_linear2idx(rhs_batch_tensor, start);

    for (int i = start; i < end; i++) {
      size_t out_batch_offset =
          tensor_idx2offset(out_batch_tensor, out_batch_idx);
      size_t lhs_batch_offset =
          tensor_idx2offset(lhs_batch_tensor, lhs_batch_idx);
      size_t rhs_batch_offset =
          tensor_idx2offset(rhs_batch_tensor, rhs_batch_idx);

      T* OUT = &tensor_data<T>(out)[out_batch_offset];
      T* LHS = &tensor_data<T>(lhs)[lhs_batch_offset];
      T* RHS = &tensor_data<T>(rhs)[rhs_batch_offset];

#define BLOCK_SIZE 64
#pragma omp parallel for collapse(2)
      for (int ii = 0; ii < M; ii += BLOCK_SIZE) {
        for (int jj = 0; jj < N; jj += BLOCK_SIZE) {
          for (int kk = 0; kk < K; kk += BLOCK_SIZE) {
            for (int i = ii;
                 i < ((ii + BLOCK_SIZE) > M ? M : (ii + BLOCK_SIZE)); i++) {
              for (int k = kk;
                   k < ((kk + BLOCK_SIZE) > K ? K : (kk + BLOCK_SIZE)); k++) {
                float r = LHS[i * LHS_M_STRIDE + k * LHS_K_STRIDE];
                for (int j = jj;
                     j < ((jj + BLOCK_SIZE) > N ? N : (jj + BLOCK_SIZE)); j++) {
                  OUT[i * OUT_M_STRIDE + j * OUT_N_STRIDE] +=
                      r * RHS[k * RHS_K_STRIDE + j * RHS_N_STRIDE];
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

void tensor_matmul_dispatch(CPUTensor& out, CPUTensor& lhs, CPUTensor& rhs) {
  switch (out.dtype) {
    case DT_F32:
      tensor_matmul_impl<float>(out, lhs, rhs);
      break;
    case DT_F64:
      tensor_matmul_impl<double>(out, lhs, rhs);
      break;
    case DT_I32:
      tensor_matmul_impl<int32_t>(out, lhs, rhs);
      break;
    case DT_I64:
      tensor_matmul_impl<int64_t>(out, lhs, rhs);
      break;
    default:
      throw std::runtime_error("Unsupported data type for matmul.");
  }
}
#endif  // __BLAS_H__
