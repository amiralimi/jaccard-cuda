#pragma once

__device__ void reduce_sum(int *const __restrict__ arr0, int *const __restrict__ arr1, int size);

void compress_1bit(
    const int *const __restrict__ src,
    unsigned int *const __restrict__ dst,
    int rows, int cols);

void calculate_similarity(
    float *const __restrict__ intersections,
    float *const __restrict__ unions,
    int n_rows,
    unsigned int window_size);

template <typename T,
std::enable_if_t<
    std::is_same_v<T, int> || std::is_same_v<T, unsigned int>, bool> = true
>
void fill_intersection_union(
    const T *const __restrict__ a,
    int n_rows,
    int n_cols,
    unsigned int window_size,
    float *const __restrict__ intersections,
    float *const __restrict__ unions);

void jaccard_similarity(
    const int *const __restrict__ a,
    int n_rows,
    int n_cols,
    unsigned int window_size,
    float *const __restrict__ results, 
    bool compress = false);
