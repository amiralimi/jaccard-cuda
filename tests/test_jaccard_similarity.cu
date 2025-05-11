#include <iostream>
#include <vector>
#include <cassert>

#include "../src/jaccard.cuh"
#include "test_utils.cuh"


void calculate_jaccard_similarity(
    const int *const __restrict__ a,
    const int n_rows,
    const int n_cols,
    const int window_size,
    float *const __restrict__ results)
{
    for (int i = 0; i < n_rows; i++)
    {
        for (int j = i + 1; j < i + 1 + window_size; j++)
        {
            if (j < n_rows)
            {
                float intersection = 0, union_set = 0;
                for (int col = 0; col < n_cols; col++)
                {
                    intersection += a[i * n_cols + col] & a[j * n_cols + col];
                    union_set += a[i * n_cols + col] | a[j * n_cols + col];
                }
                if (union_set > 0)
                    results[i * window_size + (j - i - 1)] = intersection / union_set;
                else
                    results[i * window_size + (j - i - 1)] = 0.0f;
            } else
            {
                results[i * window_size + (j - i - 1)] = 0.0f;
            }
        }
    }
}

void test_jaccard_similarity_kernel(int rows, int columns, int window_size, int seed, bool only_bench = false)
{
    std::cout << "Testing Jaccard Similarity Kernel with rows=" << rows
              << ", columns=" << columns
              << ", window_size=" << window_size
              << ", seed=" << seed << "\n";
    std::vector<int> h_in = make_random_matrix<int>(rows * columns, [](int x) { return (int)(x % 2); }, seed);

    int *d_in = h2d(h_in);

    INIT_CUDA_ARRAY(float, d_results, rows * window_size);
    INIT_CUDA_ARRAY(float, d_results_compressed, rows * window_size);

    BENCH(jaccard_similarity(d_in, rows, columns, window_size, d_results));
    BENCH(jaccard_similarity(d_in, rows, columns, window_size, d_results_compressed, true));

    if (only_bench)
    {
        cudaFree(d_in);
        cudaFree(d_results);
        cudaFree(d_results_compressed);
        std::cout << "Benchmarked\n\n";
        return;
    }

    std::vector<float> expected_results(rows * window_size, 0.0f);
    calculate_jaccard_similarity(h_in.data(), rows, columns, window_size, expected_results.data());

    assert_allclose(d_results, expected_results, 1e-5f, "Jaccard Similarity Results Uncompressed");
    assert_allclose(d_results_compressed, expected_results, 1e-5f, "Jaccard Similarity Results Compressed");

    std::cout << "Passed\n\n";
    cudaFree(d_in);
    cudaFree(d_results);
    cudaFree(d_results_compressed);
}

void test_jaccard_similarity(int seed = 42)
{
    // test_jaccard_similarity_kernel(64, 64, 64, seed);
    test_jaccard_similarity_kernel(128, 128, 128, seed);
    test_jaccard_similarity_kernel(256, 256, 256, seed);
    test_jaccard_similarity_kernel(16384, 28672, 64, seed, true);
    test_jaccard_similarity_kernel(16384, 28672, 128, seed, true);
    test_jaccard_similarity_kernel(16384, 28672, 256, seed, true);
}