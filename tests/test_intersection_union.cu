#include <iostream>
#include <vector>
#include <cassert>
#include "../src/jaccard.cuh"
#include "test_utils.cuh"

void calculate_intersection_union(
    const int *const __restrict__ a,
    const int n_rows,
    const int n_cols,
    const int window_size,
    float *const __restrict__ intersections,
    float *const __restrict__ unions)
{
    for (int i = 0; i < n_rows; i++)
    {
        for (int j = i + 1; j < i + 1 + window_size; j++)
        {
            for (int col = 0; col < n_cols; col++)
            {
                int intersection = 0, union_set = 0;
                if (j < n_rows)
                {
                    intersection = a[i * n_cols + col] & a[j * n_cols + col];
                    union_set = a[i * n_cols + col] | a[j * n_cols + col];
                } else
                {
                    intersection = a[i * n_cols + col] & 0;
                    union_set = a[i * n_cols + col] | 0;
                }
                intersections[i * window_size + (j - i - 1)] += intersection;
                unions[i * window_size + (j - i - 1)] += union_set;
            }
        }
    }
}

#define COMPRESS_AND_CALCULATE(a, compressed, n_rows, n_cols, window_size, intersections, unions)       \
    compress_1bit(a, compressed, n_rows, n_cols);                                                       \
    fill_intersection_union(compressed, n_rows, n_cols / 32, window_size, intersections, unions);


void test_intersection_union_kernel(int rows, int columns, int window_size, int seed, bool only_bench = false)
{
    std::cout << "test_intersection_union_kernel(rows=" << rows
              << ", columns=" << columns
              << ", window_size=" << window_size
              << ", seed=" << seed << ") ... \n";
    std::vector<int> h_in = make_random_matrix<int>(rows * columns, [](int x) { return (int)(x % 2); }, seed);

    int *d_in = h2d(h_in);

    float *intersections;
    float *unions;
    float *intersections_compressed;
    float *unions_compressed;
    CHECK_CUDA(cudaMalloc(&intersections, rows * window_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&unions, rows * window_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&intersections_compressed, rows * window_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&unions_compressed, rows * window_size * sizeof(float)));

    unsigned int *compressed;
    CHECK_CUDA(cudaMalloc(&compressed, rows * (columns / 32) * sizeof(unsigned int)));

    BENCH(fill_intersection_union(d_in, rows, columns, window_size, intersections, unions));
    BENCH(
        COMPRESS_AND_CALCULATE(d_in, compressed, rows, columns, window_size, intersections_compressed, unions_compressed)
    );

    if (only_bench)
    {
        cudaFree(d_in);
        cudaFree(intersections);
        cudaFree(unions);
        cudaFree(intersections_compressed);
        cudaFree(unions_compressed);
        cudaFree(compressed);
        std::cout << "Benchmarked\n\n";
        return;
    }
    
    std::vector<float> h_intersections = d2h(intersections, rows * window_size);
    std::vector<float> h_unions = d2h(unions, rows * window_size);

    std::vector<float> h_intersections_compressed = d2h(intersections_compressed, rows * window_size);
    std::vector<float> h_unions_compressed = d2h(unions_compressed, rows * window_size);

    std::vector<float> ref_intersections(rows * window_size, 0.0f);
    std::vector<float> ref_unions(rows * window_size, 0.0f);
    calculate_intersection_union(h_in.data(), rows, columns, window_size, ref_intersections.data(), ref_unions.data());
    assert_allclose(intersections, ref_intersections, 1e-5, "Intersections");
    assert_allclose(unions, ref_unions, 1e-5, "Unions");

    assert_allclose(intersections_compressed, ref_intersections, 1e-5, "Compressed Intersections");
    assert_allclose(unions_compressed, ref_unions, 1e-5, "Compressed Unions");

    std::cout << "Passed\n\n";
    cudaFree(d_in);
    cudaFree(intersections);
    cudaFree(unions);
    cudaFree(intersections_compressed);
    cudaFree(unions_compressed);
    cudaFree(compressed);
}

void test_intersection_union(int seed = 42)
{
    // test_intersection_union_kernel(64, 64, 64, seed);
    test_intersection_union_kernel(64, 128, 64, seed);
    test_intersection_union_kernel(64, 256, 64, seed);
    // test_intersection_union_kernel(128, 64, 64, seed);
    // test_intersection_union_kernel(256, 64, 64, seed);
    // test_intersection_union_kernel(128, 64, 128, seed);
    // test_intersection_union_kernel(256, 64, 128, seed);
    test_intersection_union_kernel(128, 2048, 64, seed);
    // test_intersection_union_kernel(256, 256, 64, seed);
    // test_intersection_union_kernel(128, 128, 128, seed);
    // test_intersection_union_kernel(256, 256, 128, seed);
    test_intersection_union_kernel(16384, 28672, 64, seed, true);
    // test_intersection_union_kernel(16384, 28672, 128, seed, true);
    // test_intersection_union_kernel(16384, 28672, 256, seed, true);
}
