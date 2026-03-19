#include "blas.h"

void cpu_tensor_matmul(CPUTensor out, CPUTensor lhs, CPUTensor rhs) {
  tensor_matmul_dispatch(out, lhs, rhs);
}