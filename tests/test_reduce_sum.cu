#include <iostream>
#include <vector>
#include <cassert>
#include "../src/jaccard.cuh"
#include "test_utils.cuh"

__global__ void reduce_sum_wrapper(int* d_in, int n)
{
    extern __shared__ int smem[];
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int lid = threadIdx.x;

    smem[lid] = (gid < n) ? d_in[gid] : 0;
    __syncthreads();

    reduce_sum(smem, blockDim.x);

    if (lid == 0) {
        d_in[blockIdx.x] = smem[0];
    }
}

void test_reduce_sum_kernel(int n, int seed)
{
    std::cout << "Testing reduce_sum_kernel with n=" << n
              << ", seed=" << seed << "\n";
    int blockSize = 1;
    while (blockSize < n) blockSize <<= 1;

    std::vector<int> h_in = make_random_matrix<int>(n, [](int x) { return (x % 100); }, seed);

    int* d_in = h2d(h_in);

    size_t shmem = blockSize * sizeof(int);
    reduce_sum_wrapper<<<1, blockSize, shmem>>>(d_in, n);
    CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<int> h_out = d2h(d_in, 1);

    long long total     = h_out[0];
    long long expected  = 0;
    for (int num : h_in) 
        expected += num;

    assert(total == expected);

    std::cout << "Passed\n\n";

    cudaFree(d_in);
}

void test_reduce_sum(int seed = 42)
{
    test_reduce_sum_kernel(1, seed);
    test_reduce_sum_kernel(8, seed);
    test_reduce_sum_kernel(32, seed);
    test_reduce_sum_kernel(128, seed);
    test_reduce_sum_kernel(256, seed); 
    test_reduce_sum_kernel(512, seed);
    test_reduce_sum_kernel(1024, seed);

    std::cout << "All reduce_sum tests passed\n";
}