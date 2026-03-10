#include "myrustllm-cpu.h"
#include "tensor.h"
#include "init.h"
#include <omp.h>

void cpu_tensor_init_fill_f32_(CPUTensor tensor, float val)
{
    size_t n_elements = tensor_numel(tensor);
#pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        int num_threads = omp_get_num_threads();
        int chunk_size = n_elements / num_threads;
        int start = thread_id * chunk_size;
        int end = (thread_id == num_threads - 1) ? n_elements : start + chunk_size;

        std::vector<size_t> idx = tensor_linear2idx(tensor, start);

        for (int i = start; i < end; i++)
        {
            size_t offset = tensor_idx2offset(tensor, idx);
            TENSOR_PTR(tensor.data, float, offset) = val;
            tensor_next_idx(tensor, idx);
        }
    }
}

void cpu_tensor_init_normal_f32_(CPUTensor tensor, float mean, float std, uint64_t seed)
{
    size_t n_elements = tensor_numel(tensor);
#pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        int num_threads = omp_get_num_threads();
        int chunk_size = n_elements / num_threads;
        int start = thread_id * chunk_size;
        int end = (thread_id == num_threads - 1) ? n_elements : start + chunk_size;

        std::vector<size_t> idx = tensor_linear2idx(tensor, start);

        for (int i = start; i < end; i++)
        {
            size_t offset = tensor_idx2offset(tensor, idx);
            TENSOR_PTR(tensor.data, float, offset) = normal(seed, i, mean, std);
            tensor_next_idx(tensor, idx);
        }
    }
}