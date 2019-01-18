#include <iota.hpp>

__global__ void global_iota(int size, int *vec) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < size) {
        vec[tid] = tid;
    }
}

__host__ void iota(int size, int *vec) {
    constexpr int threads = 1024;
    const int blocks = (size + threads - 1) / threads;

    int *cuda_vec;
    cudaMalloc(&cuda_vec, sizeof(int) * size);

    global_iota<<<blocks, threads>>>(size, cuda_vec);

    cudaMemcpy(vec, cuda_vec, sizeof(int) * size, cudaMemcpyDeviceToHost);
    cudaFree(&cuda_vec);
}