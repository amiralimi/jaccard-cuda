#include "jaccard.cuh"

#include <stdio.h>

int main(){
    int n_rows = 128;
    int n_cols = 1024;
    int window_size = 16;
    int *a_h = new int[n_rows * n_cols];
    // Initialize matrix a with some values
    for (int i = 0; i < n_rows * n_cols; i++) {
        a_h[i] = rand() % 2;
    }
    int *a_d;
    cudaMalloc(&a_d, n_rows * n_cols * sizeof(int));
    cudaMemcpy(a_d, a_h, n_rows * n_cols * sizeof(int), cudaMemcpyHostToDevice);
    
    float* d_jaccard_similarities = jaccard_similarity(a_d, n_rows, n_cols, window_size);
    // Copy results back to host
    float *h_jaccard_similarities = (float *)malloc(n_rows * window_size * sizeof(float));
    cudaMemcpy(h_jaccard_similarities, d_jaccard_similarities, n_rows * window_size * sizeof(float), cudaMemcpyDeviceToHost);
    // Print results
    for (int i = 0; i < n_rows; i++) {
        printf("Row %d:\n", i);
        for (int j = 0; j < window_size; j++) {
            printf("%f ", h_jaccard_similarities[i * window_size + j]);
        }
        printf("\n");
        printf("\n");
    }

    cudaFree(a_d);
    cudaFree(d_jaccard_similarities);
    return 0;
}
