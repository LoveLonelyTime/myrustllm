#ifndef MATH_H
#define MATH_H

inline size_t linear2tensor(
    size_t linear_idx,
    const size_t *strides,
    const size_t *shape,
    size_t n_dims)
{
    size_t offset = 0;

    for (size_t dim_r = 0; dim_r < n_dims; dim_r++)
    {
        size_t dim = n_dims - dim_r - 1;
        size_t dim_size = shape[dim];
        size_t dim_idx = linear_idx % dim_size;

        linear_idx /= dim_size;

        offset += dim_idx * strides[dim];
    }

    return offset;
}

#endif // MATH_H