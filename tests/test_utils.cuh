#pragma once
#include <algorithm>
#include <cuda_runtime.h>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstring>
#include <ctime>
#include <iostream>
#include <type_traits>
#include <vector>

#include "../src/utils.cuh"

template <typename T>
std::vector<T> d2h(const T *d_ptr, std::size_t n)
{
    std::vector<T> h(n);
    CHECK_CUDA(cudaMemcpy(h.data(), d_ptr, n * sizeof(T), cudaMemcpyDeviceToHost));
    return h;
}

template <typename T>
T *h2d(const std::vector<T> &h)
{
    INIT_CUDA_ARRAY(T, d_ptr, h.size());
    CHECK_CUDA(cudaMemcpy(d_ptr, h.data(),
                          h.size() * sizeof(T),
                          cudaMemcpyHostToDevice));
    return d_ptr;
}

template <typename T>
void assert_allclose(const T *d_ptr,
                     const std::vector<T> &ref,
                     double tol = 1e-5,
                     const char *tag = "")
{
    auto h = d2h(d_ptr, ref.size());

    for (std::size_t i = 0; i < ref.size(); ++i)
    {
        if constexpr (std::is_floating_point_v<T>)
        {
            // std::cout << "Comparing index " << i << ": "
            //   << h[i] << " vs " << ref[i] << " " << (int)(std::fabs(h[i] - ref[i]) < tol) << "\n" ;
            assert(std::fabs(h[i] - ref[i]) < tol);
        }
        else
        {
            // std::cout << "Comparing index " << i << ": "
            //   << h[i] << " vs " << ref[i] << "\n";
            assert(h[i] == ref[i]);
        }
    }
    if (std::strlen(tag))
        std::cout << tag << " ✔︎\n";
}

template <typename T>
std::vector<std::vector<T>>
reshape(const std::vector<T> &flat, std::size_t rows, std::size_t cols)
{
    assert(rows * cols == flat.size());
    std::vector<std::vector<T>> mat(rows, std::vector<T>(cols));
    for (std::size_t r = 0; r < rows; ++r)
        std::copy_n(flat.data() + r * cols, cols, mat[r].begin());
    return mat;
}

template <typename T, typename Transform>
std::vector<T>
make_random_matrix(std::size_t size,
                   Transform transform,
                   long seed = -1)
{
    if (seed >= 0)
        std::srand(static_cast<unsigned>(seed));
    else
        std::srand(static_cast<unsigned>(std::time(nullptr)));

    std::vector<T> out(size);
    for (auto &v : out)
        v = static_cast<T>(transform(std::rand()));

    return out;
}

#define BENCH(...)                                                               \
    do                                                                           \
    {                                                                            \
        auto _bench_start =                                                      \
            std::chrono::high_resolution_clock::now();                           \
        __VA_ARGS__;                                                             \
        CHECK_CUDA(cudaDeviceSynchronize());                                     \
        auto _bench_end =                                                        \
            std::chrono::high_resolution_clock::now();                           \
        double _bench_ms =                                                       \
            std::chrono::duration<double, std::milli>(_bench_end - _bench_start) \
                .count();                                                        \
        std::cout << #__VA_ARGS__                                                \
                  << " took " << _bench_ms << " ms\n";                           \
    } while (0)
