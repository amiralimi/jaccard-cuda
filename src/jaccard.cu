#include <iostream>
#include <stdexcept>
#include <type_traits>

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

__global__ void compress_1bit_kernel(
    const int *const __restrict__ src,
    unsigned int *const __restrict__ dst,
    int number_of_elements)
{
    constexpr unsigned FULL_MASK = 0xffffffff;
    const int col = threadIdx.x; 
    const int warp_idx = blockIdx.x * blockDim.x / 32; // Each warp processes 32 threads

    int src_idx = warp_idx * 32 + col; 
    
    u_int32_t x = (src_idx < number_of_elements) ? __ldg(&src[src_idx]) : 0;
    unsigned int packed = __ballot_sync(FULL_MASK, x != 0);
    if (col % 32 == 0)
    {
        dst[warp_idx + col / 32] = packed;
    }
}

void compress_1bit(
    const int *const __restrict__ src,
    unsigned int *const __restrict__ dst,
    int rows, int cols)
{
    int block_size = 128; // 4 warps
    int elem_count = rows * ceil((float)cols / 32);
    int grid_size = ceil((float)elem_count / (block_size / 32)); 
    std::cout << "compress_1bit: grid_size = " << grid_size << ", block_size = " << block_size << ", elem_count = " << elem_count << "\n";
    compress_1bit_kernel<<<grid_size, block_size>>>(src, dst, rows *cols);
    CHECK_CUDA(cudaDeviceSynchronize());
}

template <typename T, unsigned int WINDOW_SIZE, unsigned int BLOCK_SIZE>
__global__ void fill_intersection_union_kernel(
    const T *const __restrict__ a,
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
            if constexpr (std::is_signed_v<T>) {
                local_intersection[col] = (op1_block[col] & shared_a[j][col]);
                local_union[col] = (op1_block[col] | shared_a[j][col]);
            } else {
                local_intersection[col] = __popc(op1_block[col] & shared_a[j][col]);
                local_union[col] = __popc(op1_block[col] | shared_a[j][col]);
            }
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

const int MAX_WINDOW_SIZE = 64;
const int MAX_BLOCK_SIZE = 64;

template <typename T,
    std::enable_if_t<
        std::is_same_v<T, int> || std::is_same_v<T, unsigned int>, bool>
    >
void fill_intersection_union(
    const T *const __restrict__ a,
    int n_rows,
    int n_cols,
    unsigned int window_size,
    float *const __restrict__ intersections,
    float *const __restrict__ unions)
{
    int num_row_blocks = ceil((float)n_rows / MAX_WINDOW_SIZE);
    int num_column_blocks = ceil((float)n_cols / 32 / MAX_BLOCK_SIZE);

    // since the shared memory will run out on block sizes larger than 64x64, we need to split the window size into multiple blocks
    // and calculate the intersections and unions for each block
    int num_windows = ceil((float)window_size / MAX_WINDOW_SIZE);

    // grid dims
    dim3 grid_dim(num_row_blocks, num_column_blocks, num_windows);
    dim3 block_dim(MAX_BLOCK_SIZE);

    fill_intersection_union_kernel<T, MAX_WINDOW_SIZE, MAX_BLOCK_SIZE><<<grid_dim, block_dim>>>(a, n_rows, n_cols / 32, window_size, intersections, unions);

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
    float *const __restrict__ results, 
    bool compress)
{
    float *intersections;
    CHECK_CUDA(cudaMalloc(&intersections, n_rows * window_size * sizeof(float)));
    if (compress)
    {
        unsigned int* compressed;
        CHECK_CUDA(cudaMalloc(&compressed, n_rows * (n_cols / 32) * sizeof(unsigned int)));
        compress_1bit(a, compressed, n_rows, n_cols);
        fill_intersection_union(compressed, n_rows, n_cols / 32, window_size, intersections, results);
    } else {
        fill_intersection_union(a, n_rows, n_cols, window_size, intersections, results);
    }
    calculate_similarity(intersections, results, n_rows, window_size);
    CHECK_CUDA(cudaFree(intersections));
    CHECK_CUDA(cudaDeviceSynchronize());
}
