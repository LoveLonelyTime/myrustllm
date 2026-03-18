#include "reduce_ops.h"

struct ReduceAddOp {
  template <typename T>
  static T init() {
    return T(0);
  }

  template <typename T>
  static T apply(T a, T b) {
    return a + b;
  }
};


void cpu_tensor_reduce_add(CPUTensor lhs, CPUTensor rhs, size_t batch_dims) {
  tensor_reduce_op_dispatch<ReduceAddOp>(lhs, rhs, batch_dims);
}