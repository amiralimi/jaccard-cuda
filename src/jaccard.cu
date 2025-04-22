#include <iostream>
#include <stdexcept>

#include "jaccard.cuh"
#include "utils.cuh"

__device__ void reduce_sum(int *const __restrict__ arr, int size)
{
    unsigned int tid = threadIdx.x;
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            arr[tid] += arr[tid + stride];
        }
        __syncthreads();
    }
}

template <unsigned int WINDOW_SIZE, unsigned int BLOCK_SIZE>
__global__ void fill_intersection_union_kernel(
    const int *const __restrict__ a,
    const int n_rows,
    const int n_cols,
    const int window_size,
    float *const __restrict__ intersections,
    float *const __restrict__ unions)
{
    __shared__ int shared_a[WINDOW_SIZE * 2][BLOCK_SIZE];
    __shared__ int op1_block[BLOCK_SIZE];
    __shared__ int local_intersection[BLOCK_SIZE];
    __shared__ int local_union[BLOCK_SIZE];

    int global_row = blockIdx.x * WINDOW_SIZE;
    if (global_row >= n_rows)
    {
        return;
    }
    // we added z dimension to grid to split the window size into multiple blocks
    int load_start = global_row + blockIdx.z * WINDOW_SIZE;

    int global_col = blockIdx.y * BLOCK_SIZE + threadIdx.x;
    int col = threadIdx.x;
    if (global_col >= n_cols)
    {
        return;
    }

    // Load data into shared memory
    for (int i = load_start; i < load_start + 2 * WINDOW_SIZE; i++)
    {
        if (i < n_rows)
            shared_a[i - load_start][col] = a[i * n_cols + global_col];
        else
            shared_a[i - load_start][col] = 0;
    }
    __syncthreads();

    // Calculate intersection and union
    for (int i = 0; i < WINDOW_SIZE; i++)
    {
        // since we added the z dimension and split the window into multiple blocks, only for the first block
        // we have the first op (source row) loaded in shared memory
        // for the rest of the blocks, we load the first op from global memory
        // int *op1;
        if (blockIdx.z == 0)
        {
            op1_block[col] = shared_a[i][col];
        }
        else
        {
            op1_block[col] = a[(i + global_row) * n_cols + global_col];
            __syncthreads();
        }
        for (int j = i + 1; j < i + 1 + WINDOW_SIZE; j++)
        {
            local_intersection[col] = (op1_block[col] & shared_a[j][col]);
            local_union[col] = (op1_block[col] | shared_a[j][col]);
            // reduce intersection and union counts
            reduce_sum(local_intersection, BLOCK_SIZE);
            reduce_sum(local_union, BLOCK_SIZE);
            // Store results
            if (threadIdx.x == 0)
            {
                atomicAdd(&intersections[(global_row + i) * window_size + (j - i - 1) + blockIdx.z * WINDOW_SIZE], (float)local_intersection[0]);
                atomicAdd(&unions[(global_row + i) * window_size + (j - i - 1) + blockIdx.z * WINDOW_SIZE], (float)local_union[0]);
            }
            __syncthreads();
        }
    }
}

// #define INSTANTIATE_INTERSECTION_UNION_KERNEL(WINDOW_SIZE, BLOCK_SIZE)                \
//     template __global__ void fill_intersection_union_kernel<WINDOW_SIZE, BLOCK_SIZE>( \
//         const int *const __restrict__ a,                                              \
//         const int n_rows,                                                             \
//         const int n_cols,                                                             \
//         float *const __restrict__ intersections,                                      \
//         float *const __restrict__ unions);

// #ifdef BUILD_TESTS
// INSTANTIATE_INTERSECTION_UNION_KERNEL(64, 64)
// INSTANTIATE_INTERSECTION_UNION_KERNEL(32, 32)
// #endif // BUILD_TESTS

const int MAX_WINDOW_SIZE = 64;
const int MAX_BLOCK_SIZE = 64;

void fill_intersection_union(
    const int *const __restrict__ a,
    int n_rows,
    int n_cols,
    unsigned int window_size,
    float *const __restrict__ intersections,
    float *const __restrict__ unions)
{
    int num_row_blocks = ceil((float)n_rows / MAX_WINDOW_SIZE);
    int num_column_blocks = ceil((float)n_cols / MAX_BLOCK_SIZE);

    // since the shared memory will run out on block sizes larger than 64x64, we need to split the window size into multiple blocks
    // and calculate the intersections and unions for each block
    int num_windows = ceil((float)window_size / MAX_WINDOW_SIZE);

    // grid dims
    dim3 grid_dim(num_row_blocks, num_column_blocks, num_windows);
    dim3 block_dim(MAX_BLOCK_SIZE);

    fill_intersection_union_kernel<MAX_WINDOW_SIZE, MAX_BLOCK_SIZE><<<grid_dim, block_dim>>>(a, n_rows, n_cols, window_size, intersections, unions);

    CHECK_CUDA(cudaDeviceSynchronize());
}

void __global__ calculate_similarity_kernel(
    float *const __restrict__ intersections,
    float *const __restrict__ unions,
    int num_elems)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elems)
    {
        unions[idx] = ((unions[idx] == 0 | intersections[idx] == 0) ? 0 : (float)intersections[idx] / (float)unions[idx]);
    }
}

void calculate_similarity(
    float *const __restrict__ intersections,
    float *const __restrict__ unions,
    int n_rows,
    unsigned int window_size)
{
    int num_elems = n_rows * window_size;
    int block_size = 1024;
    int num_blocks = (num_elems + block_size - 1) / block_size;

    calculate_similarity_kernel<<<num_blocks, block_size>>>(intersections, unions, num_elems);

    CHECK_CUDA(cudaDeviceSynchronize());
}

void jaccard_similarity(
    const int *const __restrict__ a,
    int n_rows,
    int n_cols,
    unsigned int window_size,
    float *const __restrict__ results)
{
    float *intersections;
    CHECK_CUDA(cudaMalloc(&intersections, n_rows * window_size * sizeof(float)));
    fill_intersection_union(a, n_rows, n_cols, window_size, intersections, results);
    calculate_similarity(intersections, results, n_rows, window_size);
    CHECK_CUDA(cudaFree(intersections));
    CHECK_CUDA(cudaDeviceSynchronize());
}