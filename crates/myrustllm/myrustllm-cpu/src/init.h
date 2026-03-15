#ifndef __INIT_H__
#define __INIT_H__

#include <cmath>
#include <cstdint>

const uint32_t PHILOX_M0 = 0xD2511F53;
const uint32_t PHILOX_M1 = 0xCD9E8D57;
const uint32_t PHILOX_W0 = 0x9E3779B9;
const uint32_t PHILOX_W1 = 0xBB67AE85;

inline void mulhilo(uint32_t a, uint32_t b, uint32_t &hi, uint32_t &lo)
{
    uint64_t product = (uint64_t)a * (uint64_t)b;
    hi = product >> 32;
    lo = product & 0xffffffff;
}

inline void philox_round(uint32_t counter[4], uint32_t key[2])
{
    uint32_t hi0, lo0;
    uint32_t hi1, lo1;

    mulhilo(PHILOX_M0, counter[0], hi0, lo0);
    mulhilo(PHILOX_M1, counter[2], hi1, lo1);

    uint32_t c0 = hi1 ^ counter[1] ^ key[0];
    uint32_t c1 = lo1;
    uint32_t c2 = hi0 ^ counter[3] ^ key[1];
    uint32_t c3 = lo0;

    counter[0] = c0;
    counter[1] = c1;
    counter[2] = c2;
    counter[3] = c3;
}

inline void philox(uint64_t seed, uint64_t index, uint32_t out[4])
{
    uint32_t counter[4] = {
        (uint32_t)index,
        (uint32_t)(index >> 32),
        0,
        0};

    uint32_t key[2] = {
        (uint32_t)seed,
        (uint32_t)(seed >> 32)};

    for (int i = 0; i < 10; i++)
    {
        philox_round(counter, key);
        key[0] += PHILOX_W0;
        key[1] += PHILOX_W1;
    }

    out[0] = counter[0];
    out[1] = counter[1];
    out[2] = counter[2];
    out[3] = counter[3];
}

inline float uniform(uint64_t seed, uint64_t index)
{
    uint32_t r[4];
    philox(seed, index, r);
    return (float)r[0] / (float)UINT32_MAX;
}

inline float normal(uint64_t seed, uint64_t index, float mean, float std)
{
    float u1 = uniform(seed, index * 2);
    float u2 = uniform(seed, index * 2 + 1);

    float r = std::sqrt(-2.0f * std::log(u1));
    float theta = 2.0f * M_PI * u2;

    return mean + std * r * std::cos(theta);
}

// template<typename T>
// void tensor_init_fill_impl(CPUTensor &tensor, T val)
// {
//     size_t n_elements = tensor_numel(tensor);
// #pragma omp parallel
//     {
//         int thread_id = omp_get_thread_num();
//         int num_threads = omp_get_num_threads();
//         int chunk_size = n_elements / num_threads;
//         int start = thread_id * chunk_size;
//         int end = (thread_id == num_threads - 1) ? n_elements : start + chunk_size;

//         std::vector<size_t> idx = tensor_linear2idx(tensor, start);

//         for (int i = start; i < end; i++)
//         {
//             size_t offset = tensor_idx2offset(tensor, idx);
//             tensor_data<T>(tensor)[offset] = val;
//             tensor_next_idx(tensor, idx);
//         }
//     }
// }

// template<typename T>
// void tensor_init_normal_impl(CPUTensor tensor, float mean, float std, uint64_t seed)
// {
//     size_t n_elements = tensor_numel(tensor);
// #pragma omp parallel
//     {
//         int thread_id = omp_get_thread_num();
//         int num_threads = omp_get_num_threads();
//         int chunk_size = n_elements / num_threads;
//         int start = thread_id * chunk_size;
//         int end = (thread_id == num_threads - 1) ? n_elements : start + chunk_size;

//         std::vector<size_t> idx = tensor_linear2idx(tensor, start);

//         for (int i = start; i < end; i++)
//         {
//             size_t offset = tensor_idx2offset(tensor, idx);
//             TENSOR_PTR(tensor.data, float, offset) = normal(seed, i, mean, std);
//             tensor_next_idx(tensor, idx);
//         }
//     }
// }

#endif // __INIT_H__