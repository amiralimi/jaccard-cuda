#pragma once

#define CHECK_CUDA(call)                                               \
    do                                                                 \
    {                                                                  \
        cudaError_t _e = (call);                                       \
        if (_e != cudaSuccess)                                         \
        {                                                              \
            std::cerr << "CUDA error: " << cudaGetErrorString(_e)      \
                      << " (" << __FILE__ << ':' << __LINE__ << ")\n"; \
            std::exit(EXIT_FAILURE);                                   \
        }                                                              \
    } while (0)

#define INIT_CUDA_ARRAY(type, name, size)                              \
    type *name;                                                        \
    CHECK_CUDA(cudaMalloc((void **)&name, size * sizeof(type)));       \
    CHECK_CUDA(cudaMemset(name, 0, size * sizeof(type)));              \
    CHECK_CUDA(cudaDeviceSynchronize());
