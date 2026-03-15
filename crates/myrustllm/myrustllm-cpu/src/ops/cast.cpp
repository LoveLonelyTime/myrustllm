#include "cast.h"
#include <stdexcept>

#define REGISTER_CAST(O, L) tensor_cast_table[O][L] = tensor_cast_dispatch<O, L>;

using CastFn = void (*)(CPUTensor &, CPUTensor &);

static CastFn tensor_cast_table[DT_COUNT][DT_COUNT] = {nullptr};

static struct CastTableInitializer
{
    CastTableInitializer()
    {
        REGISTER_CAST(DT_F32, DT_F32)
        REGISTER_CAST(DT_F32, DT_F64)
        REGISTER_CAST(DT_F32, DT_I32)
        REGISTER_CAST(DT_F32, DT_I64)

        REGISTER_CAST(DT_F64, DT_F32)
        REGISTER_CAST(DT_F64, DT_F64)
        REGISTER_CAST(DT_F64, DT_I32)
        REGISTER_CAST(DT_F64, DT_I64)

        REGISTER_CAST(DT_I32, DT_F32)
        REGISTER_CAST(DT_I32, DT_F64)
        REGISTER_CAST(DT_I32, DT_I32)
        REGISTER_CAST(DT_I32, DT_I64)

        REGISTER_CAST(DT_I64, DT_F32)
        REGISTER_CAST(DT_I64, DT_F64)
        REGISTER_CAST(DT_I64, DT_I32)
        REGISTER_CAST(DT_I64, DT_I64)
    }
} cast_table_initializer;

void cpu_tensor_cast(CPUTensor out, CPUTensor lhs)
{
    CastFn func = tensor_cast_table[out.dtype][lhs.dtype];
    if (func)
    {
        func(out, lhs);
    }
    else
    {
        throw std::runtime_error("Unsupported type conversion.");
    }
}

void cpu_tensor_copy(CPUTensor dst, CPUTensor src)
{
    switch (dst.dtype)
    {
    case DT_F32:
        tensor_copy_impl<float>(dst, src);
        break;
    case DT_F64:
        tensor_copy_impl<double>(dst, src);
        break;
    case DT_I32:
        tensor_copy_impl<int32_t>(dst, src);
        break;
    case DT_I64:
        tensor_copy_impl<int64_t>(dst, src);
        break;
    default:
        throw std::runtime_error("Unsupported data type.");
    }
}