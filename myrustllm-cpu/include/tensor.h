#ifndef TENSOR_H
#define TENSOR_H

#include <myrustllm-cpu.h>
#include <vector>

#define TENSOR_PTR(ptr, ty, offset) reinterpret_cast<ty*>(ptr)[offset]

inline size_t tensor_numel(CPUTensor tensor)
{
    size_t numel = 1;
    for (size_t dim = 0; dim < tensor.dims; dim++)
    {
        numel *= tensor.shape[dim];
    }
    return numel;
}

inline std::vector<size_t> tensor_linear2idx(CPUTensor tensor, size_t linear_idx)
{
    std::vector<size_t> idx(tensor.dims);

    for (size_t dim = tensor.dims; dim > 0; dim--)
    {
        size_t dim_size = tensor.shape[dim - 1];
        size_t dim_idx = linear_idx % dim_size;

        linear_idx /= dim_size;

        idx[dim - 1] = dim_idx;
    }
}

inline void tensor_next_idx(CPUTensor tensor, std::vector<size_t> &idx)
{
    idx[tensor.dims - 1]++;
    for (size_t dim = tensor.dims; dim > 0; dim--)
    {
        if (idx[dim - 1] >= tensor.shape[dim - 1])
        {
            idx[dim - 1] = 0;
            idx[dim - 2]++;
        }
        else
        {
            break;
        }
    }
}

inline size_t tensor_idx2offset(CPUTensor tensor, std::vector<size_t> &idx)
{
    size_t offset = 0;
    for (size_t dim = 0; dim < tensor.dims; dim++)
    {
        offset += idx[dim] * tensor.stride[dim];
    }
    return offset;
}

#endif // TENSOR_H