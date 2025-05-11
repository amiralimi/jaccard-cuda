#include <iostream>
#include <vector>
#include <cassert>
#include "../src/jaccard.cuh"
#include "test_utils.cuh"

__global__ void reduce_sum_wrapper(int* d_in1, int* d_in2, int n)
{
    __shared__ int arr1[1024];
    __shared__ int arr2[1024];

    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int lid = threadIdx.x;

    arr1[lid] = (gid < n) ? d_in1[gid] : 0;
    arr2[lid] = (gid < n) ? d_in2[gid] : 0;
    __syncthreads();

    reduce_sum(arr1, arr2, blockDim.x);

    if (lid == 0) {
        d_in1[blockIdx.x] = arr1[0];
        d_in2[blockIdx.x] = arr2[0];
    }
}

// #define INSTANTIATE_REDUCE_SUM_WRAPPER(T)               \
//     template void reduce_sum_wrapper<T>(                \
//         int* d_in1, int* d_in2, int n);

// #ifdef BUILD_TESTS
// INSTANTIATE_REDUCE_SUM_WRAPPER(1)
// INSTANTIATE_REDUCE_SUM_WRAPPER(8)
// INSTANTIATE_REDUCE_SUM_WRAPPER(32)
// INSTANTIATE_REDUCE_SUM_WRAPPER(128)
// INSTANTIATE_REDUCE_SUM_WRAPPER(256)
// INSTANTIATE_REDUCE_SUM_WRAPPER(512)
// INSTANTIATE_REDUCE_SUM_WRAPPER(1024)
// #endif // BUILD_TESTS

void test_reduce_sum_kernel(int n, int seed)
{
    std::cout << "test_reduce_sum_kernel(n=" << n << ")\n";
    int blockSize = 1;
    while (blockSize < n) blockSize <<= 1;

    std::vector<int> h_in1 = make_random_matrix<int>(n, [](int x) { return (x % 100); }, seed);
    std::vector<int> h_in2 = make_random_matrix<int>(n, [](int x) { return (x % 100); }, seed);

    int* d_in1 = h2d(h_in1);
    int* d_in2 = h2d(h_in2);

    BENCH(reduce_sum_wrapper<<<1, blockSize>>>(d_in1, d_in2, n));
    CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<int> h_out1 = d2h(d_in1, 1);
    std::vector<int> h_out2 = d2h(d_in2, 1);

    long long total1     = h_out1[0];
    long long total2     = h_out2[0];
    long long expected1  = 0;
    long long expected2  = 0;
    for (int num : h_in1) 
        expected1 += num;
    
    for (int num : h_in2)
        expected2 += num;

    std::cout << "total1=" << total1 << " expected1=" << expected1 << "\n";
    std::cout << "total2=" << total2 << " expected2=" << expected2 << "\n";
    assert(total1 == expected1);
    assert(total2 == expected2);

    std::cout << "test_reduce_sum_single_block(n=" << n
              << ", blockSize=" << blockSize << ") ✔︎ Σ=" << total1 << " & Σ=" << total2 << "\n";

    cudaFree(d_in1);
    cudaFree(d_in2);
}

void test_reduce_sum(int seed = 42)
{
    // test_reduce_sum_kernel(1, seed);
    // test_reduce_sum_kernel(8, seed);
    test_reduce_sum_kernel(64, seed);
    // test_reduce_sum_kernel(128, seed);
    // test_reduce_sum_kernel(256, seed); 
    // test_reduce_sum_kernel(512, seed);
    // test_reduce_sum_kernel(1024, seed);

    std::cout << "All reduce_sum tests passed\n";
}
