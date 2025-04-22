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
