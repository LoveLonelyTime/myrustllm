#include "binary_ops.h"

void cpu_tensor_add(CPUTensor out, CPUTensor lhs, CPUTensor rhs)
{
    tensor_binary_op(out, lhs, rhs, std::plus<>(), out.dtype);
}