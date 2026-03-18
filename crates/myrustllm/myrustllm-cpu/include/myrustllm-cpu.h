#ifndef MYRUSTLLM_CPU_H
#define MYRUSTLLM_CPU_H

#include <stdlib.h>
#include <stdint.h>

#if defined(_WIN32)
#define __MYRUSTLLM_API__ __declspec(dlllimport)
#elif defined(__GNUC__)
#define __MYRUSTLLM_API__ __attribute__((visibility("default")))
#else
#define __MYRUSTLLM_API__
#endif

#define DT_COUNT 4
#define DT_F32 0
#define DT_F64 1
#define DT_I32 2
#define DT_I64 3

#ifdef __cplusplus
extern "C"
{
#endif

    typedef uint8_t DType;

    typedef struct
    {
        uint8_t *data;
        const size_t *shape;
        const size_t *stride;
        size_t dims;
        DType dtype;
    } CPUTensor;

    // ---------- Math ----------
    // FP32 COPY
    // void cpu_tensor_copy_f32(CPUTensor lhs, CPUTensor rhs);

    // // FP32 ADD
    // __MYRUSTLLM_API__ void cpu_tensor_add_f32(CPUTensor out, CPUTensor lhs, CPUTensor rhs);

    // // FP32 ADD_
    // __MYRUSTLLM_API__ void cpu_tensor_add_f32_(CPUTensor lhs, CPUTensor rhs);

    // // FP32 ADD_R
    // __MYRUSTLLM_API__ void cpu_tensor_add_f32_r(CPUTensor lhs, CPUTensor rhs, size_t batch_dims);

    // // FP32 MUL
    // __MYRUSTLLM_API__ void cpu_tensor_mul_f32(CPUTensor out, CPUTensor lhs, CPUTensor rhs);

    // // FP32 SUB_
    // __MYRUSTLLM_API__ void cpu_tensor_sub_f32_(CPUTensor lhs, CPUTensor rhs);

    // // FP32 MATMUL
    // __MYRUSTLLM_API__ void cpu_tensor_matmul_f32(CPUTensor out, CPUTensor lhs, CPUTensor rhs);

    // // FP32 INIT FILL
    // __MYRUSTLLM_API__ void cpu_tensor_init_fill_f32_(CPUTensor tensor, float val);

    // // FP32 INIT NORMAL
    // __MYRUSTLLM_API__ void cpu_tensor_init_normal_f32_(CPUTensor tensor, float mean, float std, uint64_t seed);

    __MYRUSTLLM_API__ void cpu_tensor_cast(CPUTensor out, CPUTensor lhs);

    __MYRUSTLLM_API__ void cpu_tensor_copy(CPUTensor dst, CPUTensor src);

    __MYRUSTLLM_API__ void cpu_tensor_add(CPUTensor out, CPUTensor lhs, CPUTensor rhs);
    __MYRUSTLLM_API__ void cpu_tensor_sub(CPUTensor out, CPUTensor lhs, CPUTensor rhs);
    __MYRUSTLLM_API__ void cpu_tensor_mul(CPUTensor out, CPUTensor lhs, CPUTensor rhs);
    __MYRUSTLLM_API__ void cpu_tensor_div(CPUTensor out, CPUTensor lhs, CPUTensor rhs);

    __MYRUSTLLM_API__ void cpu_tensor_reduce_add(CPUTensor lhs, CPUTensor rhs, size_t batch_dims);

#ifdef __cplusplus
}
#endif

#endif // MYRUSTLLM_CPU_H