#ifndef __TENSOR_H__
#define __TENSOR_H__

#include "myrustllm-cpu.h"
#include <vector>

template <uint8_t>
struct dtype_to_type;

template <>
struct dtype_to_type<DT_F32> { using type = float; };

template <>
struct dtype_to_type<DT_F64> { using type = double; };

template <>
struct dtype_to_type<DT_I32> { using type = int32_t; };

template <>
struct dtype_to_type<DT_I64> { using type = int64_t; };


template <typename T>
inline T *tensor_data(CPUTensor &tensor)
{
    return reinterpret_cast<T *>(tensor.data);
}

inline size_t tensor_numel(CPUTensor &tensor)
{
    size_t numel = 1;
    for (size_t dim = 0; dim < tensor.dims; dim++)
    {
        numel *= tensor.shape[dim];
    }
    return numel;
}

inline std::vector<size_t> tensor_linear2idx(CPUTensor &tensor, size_t linear_idx)
{
    std::vector<size_t> idx(tensor.dims);

    for (size_t dim = tensor.dims; dim > 0; dim--)
    {

        size_t dim_size = tensor.shape[dim - 1];
        size_t dim_idx = linear_idx % dim_size;

        linear_idx /= dim_size;

        idx[dim - 1] = dim_idx;
    }

    return idx;
}

inline void tensor_next_idx(CPUTensor &tensor, std::vector<size_t> &idx)
{
    if (tensor.dims == 0)
    {
        // Scalar
        return;
    }
    idx[tensor.dims - 1]++;

    // Exclude 0 dim
    for (size_t dim = tensor.dims; dim > 1; dim--)
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

inline size_t tensor_idx2offset(CPUTensor &tensor, const std::vector<size_t> &idx)
{
    size_t offset = 0;
    for (size_t dim = 0; dim < tensor.dims; dim++)
    {
        offset += idx[dim] * tensor.stride[dim];
    }
    return offset;
}

#endif // __TENSOR_H__