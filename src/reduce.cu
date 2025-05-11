#include <bits/stdc++.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>


#define THREAD_PER_BLOCK 512
#define WARP_SIZE 32

#define BENCH(...)                                                               \
    do                                                                           \
    {                                                                            \
        auto _bench_start =                                                      \
            std::chrono::high_resolution_clock::now();                           \
        __VA_ARGS__;                                                             \
        auto _bench_end =                                                        \
            std::chrono::high_resolution_clock::now();                           \
        double _bench_ms =                                                       \
            std::chrono::duration<double, std::milli>(_bench_end - _bench_start) \
                .count();                                                        \
        std::cout << #__VA_ARGS__                                                \
                  << " took " << _bench_ms << " ms\n";                           \
    } while (0)


template <unsigned int THREAD_IN_BLOCK>
__device__ void warpReduce(volatile float *cache, unsigned int tid) {
    if (THREAD_IN_BLOCK >= 64)
        cache[tid] += cache[tid + 32];
    if (THREAD_IN_BLOCK >= 32)
        cache[tid] += cache[tid + 16];
    if (THREAD_IN_BLOCK >= 16)
        cache[tid] += cache[tid + 8];
    if (THREAD_IN_BLOCK >= 8)
        cache[tid] += cache[tid + 4];
    if (THREAD_IN_BLOCK >= 4)
        cache[tid] += cache[tid + 2];
    if (THREAD_IN_BLOCK >= 2)
        cache[tid] += cache[tid + 1];
}

// @vgod
template <unsigned int THREAD_IN_BLOCK, int NUM_ELE_PER_THREAD>
__global__ void reduce1(float *d_in, float *d_out, unsigned int n) {
    __shared__ float sdata[THREAD_IN_BLOCK];

    // each thread loads NUM_PER_THREAD element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i   = blockIdx.x * (THREAD_IN_BLOCK * NUM_ELE_PER_THREAD) + threadIdx.x;

    sdata[tid] = 0;

#pragma unroll
    for (int iter = 0; iter < NUM_ELE_PER_THREAD; iter++) {
        sdata[tid] += d_in[i + iter * THREAD_IN_BLOCK];
    }

    __syncthreads();

    // do reduction in shared mem
    if (THREAD_IN_BLOCK == 1024) {
        if (tid < 512) {
            sdata[tid] += sdata[tid + 512];
        }
        __syncthreads();
    }
    if (THREAD_IN_BLOCK >= 512) {
        if (tid < 256) {
            sdata[tid] += sdata[tid + 256];
        }
        __syncthreads();
    }
    if (THREAD_IN_BLOCK >= 256) {
        if (tid < 128) {
            sdata[tid] += sdata[tid + 128];
        }
        __syncthreads();
    }
    if (THREAD_IN_BLOCK >= 128) {
        if (tid < 64) {
            sdata[tid] += sdata[tid + 64];
        }
        __syncthreads();
    }
    if (tid < 32)
        warpReduce<THREAD_IN_BLOCK>(sdata, tid);

    // write result for this block to global mem
    if (tid == 0)
        d_out[blockIdx.x] = sdata[0];
}

// @vgod
template <unsigned int THREAD_IN_BLOCK>
__device__ __forceinline__ float warpReduceSum(float sum) {
    if (THREAD_IN_BLOCK >= 32) sum += __shfl_down_sync(0xffffffff, sum, 16); // 0-16, 1-17, 2-18, etc.
    if (THREAD_IN_BLOCK >= 16) sum += __shfl_down_sync(0xffffffff, sum, 8);// 0-8, 1-9, 2-10, etc.
    if (THREAD_IN_BLOCK >= 8) sum += __shfl_down_sync(0xffffffff, sum, 4);// 0-4, 1-5, 2-6, etc.
    if (THREAD_IN_BLOCK >= 4) sum += __shfl_down_sync(0xffffffff, sum, 2);// 0-2, 1-3, 4-6, 5-7, etc.
    if (THREAD_IN_BLOCK >= 2) sum += __shfl_down_sync(0xffffffff, sum, 1);// 0-1, 2-3, 4-5, etc.
    return sum;
}
template <unsigned int THREAD_IN_BLOCK, int NUM_ELE_PER_THREAD>
__global__ void reduce2(float *d_in,float *d_out, unsigned int n){
    float sum = 0;
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (THREAD_IN_BLOCK * NUM_ELE_PER_THREAD) + threadIdx.x;
    #pragma unroll
    for(int iter=0; iter < NUM_ELE_PER_THREAD; iter++){
        sum += d_in[i + iter * THREAD_IN_BLOCK];
    }
    static __shared__ float warpLevelSums[WARP_SIZE]; 
    const int laneId = threadIdx.x % WARP_SIZE;
    const int warpId = threadIdx.x / WARP_SIZE;
    sum = warpReduceSum<THREAD_IN_BLOCK>(sum);
    if(laneId == 0 ) warpLevelSums[warpId] = sum;
    __syncthreads();
    sum = (threadIdx.x < blockDim.x / WARP_SIZE) ? warpLevelSums[laneId] : 0;
    if (warpId == 0) sum = warpReduceSum<THREAD_IN_BLOCK/WARP_SIZE>(sum); 
    if (tid == 0) d_out[blockIdx.x] = sum;
}


bool check(float *out, float *res, int n) {
    for (int i = 0; i < n; i++) {
        if (out[i] != res[i])
            return false;
    }
    return true;
}

int main() {
    const int N = 64 * 1024 * 1024;
    float *a    = (float *)malloc(N * sizeof(float));
    float *d_a;
    cudaMalloc((void **)&d_a, N * sizeof(float));

    const int NUM_BLOCK          = 1024;
    const int NUM_ELE_PER_BLOCK  = N / NUM_BLOCK;
    const int NUM_ELE_PER_THREAD = NUM_ELE_PER_BLOCK / THREAD_PER_BLOCK;
    float *out1                  = (float *)malloc(NUM_BLOCK * sizeof(float));
    float *out2                  = (float *)malloc(NUM_BLOCK * sizeof(float));
    float *d_out1, *d_out2;
    cudaMalloc((void **)&d_out1, NUM_BLOCK * sizeof(float));
    cudaMalloc((void **)&d_out2, NUM_BLOCK * sizeof(float));
    float *res = (float *)malloc(NUM_BLOCK * sizeof(float));

    for (int i = 0; i < N; i++) {
        a[i] = i % 456;
    }

    for (int i = 0; i < NUM_BLOCK; i++) {
        float cur = 0;
        for (int j = 0; j < NUM_ELE_PER_BLOCK; j++) {
            if (i * NUM_ELE_PER_BLOCK + j < N) {
                cur += a[i * NUM_ELE_PER_BLOCK + j];
            }
        }
        res[i] = cur;
    }

    cudaMemcpy(d_a, a, N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 Grid(NUM_BLOCK, 1);
    dim3 Block(THREAD_PER_BLOCK, 1);
    BENCH(reduce2<THREAD_PER_BLOCK, NUM_ELE_PER_THREAD><<<Grid, Block>>>(d_a, d_out1, N));
    BENCH(reduce2<THREAD_PER_BLOCK, NUM_ELE_PER_THREAD><<<Grid, Block>>>(d_a, d_out2, N));

    cudaMemcpy(out1, d_out1, NUM_BLOCK * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(out2, d_out2, NUM_BLOCK * sizeof(float), cudaMemcpyDeviceToHost);

    if (check(out1, res, NUM_BLOCK) && check(out2, res, NUM_BLOCK)) {
        printf("the ans is right\n");
    } else {
        printf("the ans is wrong\n");
        for (int i = 0; i < NUM_BLOCK; i++) {
            printf("%lf ", out1[i]);
            printf("%lf ", out2[i]);
        }
        printf("\n");
    }

    cudaFree(d_a);
    cudaFree(d_out1);
    cudaFree(d_out2);
}