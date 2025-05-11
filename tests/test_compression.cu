#include <iostream>
#include <vector>
#include <cassert>
#include "../src/jaccard.cuh"
#include "test_utils.cuh"

void calculate_compress_1bit(
    const int *const __restrict__ src, 
    unsigned int *const __restrict__ dst,
    const int n_rows,
    const int n_cols)
{
    for (int i = 0; i < n_rows; i++)
    {
        for (int j = 0; j < n_cols; j+=32)
        {
            // Process 32 bits at a time
            unsigned int bitset = 0;
            for (int k = 0; k < 32 && j + k < n_cols; k++)
            {
                bitset |= (src[i * n_cols + j + k] > 0 ? 1 : 0) << k;
            }
            dst[i * n_cols / 32 + j / 32] = bitset;
        }
    }
}

void test_compress_1bit_kernel(int n_rows, int n_cols, int seed, bool only_bench = false)
{
    std::cout << "Testing compress_1bit_kernel with n_rows=" << n_rows
              << ", n_cols=" << n_cols
              << ", seed=" << seed << "\n";
    std::vector<int> h_in = make_random_matrix<int>(n_rows * n_cols, [](int x) { return (int)(x % 2); }, seed);

    int *d_in = h2d(h_in);

    INIT_CUDA_ARRAY(unsigned int, d_out, n_rows * (n_cols / 32));

    BENCH(compress_1bit(d_in, d_out, n_rows, n_cols));

    if (only_bench)
    {
        cudaFree(d_in);
        cudaFree(d_out);
        std::cout << "Benchmarked\n\n";
        return;
    }

    // Validate the output
    std::vector<unsigned int> expected(n_rows * (n_cols / 32));
    calculate_compress_1bit(h_in.data(), expected.data(), n_rows, n_cols);
    
    assert_allclose(d_out, expected, 0, "compress_1bit output Results");

    std::cout << "Passed\n\n";

    cudaFree(d_in);
    cudaFree(d_out);
}

void test_compress_1bit(int seed = 42)
{
    test_compress_1bit_kernel(64, 128, seed);
    test_compress_1bit_kernel(128, 256, seed);
    test_compress_1bit_kernel(256, 512, seed);
    test_compress_1bit_kernel(512, 1024, seed);
    test_compress_1bit_kernel(16384, 28672, seed, true);
}
