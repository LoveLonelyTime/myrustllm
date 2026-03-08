#include <myrustllm-cuda.h>
#include <check.h>

void* cuda_tensor_alloc(size_t count) {
    void *ptr;
    CUDA_CHECK(cudaMalloc(&ptr, count));
    return ptr;
}

void cuda_tensor_destroy(void* ptr) {
    CUDA_CHECK(cudaFree(ptr));
}

void cuda_tensor_copy_from_host(void* dst, const void* src, size_t count) {
    CUDA_CHECK(cudaMemcpy(dst, src, count, cudaMemcpyHostToDevice));
}

void cuda_tensor_copy_from_device(void* dst, const void* src, size_t count) {
    CUDA_CHECK(cudaMemcpy(dst, src, count, cudaMemcpyDeviceToHost));
}