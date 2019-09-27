#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define N (2048*2048)
#define THREADS_PER_BLOCK 512

__global__ void add(int *a, int *b, int *c) {
    int index = threadIdx.x + blockInx.x * blockDim.x;
    c[index] = a[index] + b[index];
}
int main() {
    int *a, *b, *c;
    
    // host copies of variables a, b & c
    int *d_a, *d_b, *d_c;
    
    // device copies of variables a, b & c
    int size = N * sizeof(int);
    
    // Allocate space for device copies of a, b, c
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);
    
    // Setup input values  
    a = (int *) malloc(size); 
    b = (int *) malloc(size);
    c = (int *) malloc(size);
    
    for (int i = 0; i < N; i++) {
        a[i] = i;
        b[i] = -i;
    }
    
    // Copy inputs to device
    cudaMemcpy(d_a, &a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, &b, size, cudaMemcpyHostToDevice);
    
    // Launch add() kernel on GPU
    add<<<N/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_a, d_b, d_c);
    
    // Copy result back to host
    cudaError err = cudaMemcpy(&c, d_c, size, cudaMemcpyDeviceToHost);
    
    if(err!=cudaSuccess) {
        printf("CUDA error copying to Host: %s\n", cudaGetErrorString(err));
    }
    
    for (int i = 0; i < 10; i++) {
        printf("%d ", c[i]);
    }
    
    // Cleanup
    free(a); free(b); free(c)
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    
    return 0;
}
