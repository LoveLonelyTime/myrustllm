#include "binary_ops.h"

void cpu_tensor_add(CPUTensor out, CPUTensor lhs, CPUTensor rhs)
{
    tensor_binary_op(out, lhs, rhs, std::plus<>(), out.dtype);
}

void cpu_tensor_sub(CPUTensor out, CPUTensor lhs, CPUTensor rhs)
{
    tensor_binary_op(out, lhs, rhs, std::minus<>(), out.dtype);
}

void cpu_tensor_mul(CPUTensor out, CPUTensor lhs, CPUTensor rhs)
{
    tensor_binary_op(out, lhs, rhs, std::multiplies<>(), out.dtype);
}

void cpu_tensor_div(CPUTensor out, CPUTensor lhs, CPUTensor rhs)
{
    tensor_binary_op(out, lhs, rhs, std::divides<>(), out.dtype);
}
