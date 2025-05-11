#pragma once

__device__ void reduce_sum(int *const __restrict__ arr0, int *const __restrict__ arr1, int size);

void calculate_similarity(
    float *const __restrict__ intersections,
    float *const __restrict__ unions,
    int n_rows,
    unsigned int window_size);

void fill_intersection_union(
    const int *const __restrict__ a,
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
    float *const __restrict__ results);