
/*  Bill Derksen - 2/14/2021

    Example / test CUDA program. Sums two vectors.

*/

//cuda
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//std
#include <stdio.h>
#include <iostream>
#include <stdexcept>
#include <cassert>
#include <chrono>
#include <iomanip>

void printDeviceWarpSize();

// KERNEL
__global__ void sampleAddKernel(float *a, float* b, float* c)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    c[i] = a[i] + b[i];
}

// MAIN
int main()
{
    std::cout << "Starting cuda program...\n";
    printDeviceWarpSize();

    float * host_data_a,* host_data_b, *host_data_c;
    float * device_data_a,* device_data_b, *device_data_c;

    int N, nBytes;

    N = 1024;// n elements
    nBytes = N * sizeof(float);

    // Allocate host (CPU) memory
    host_data_a = (float*)malloc(nBytes);
    host_data_b = (float*)malloc(nBytes);
    host_data_c = (float*)malloc(nBytes);

    // Allocate device (GPU) memory
    try {
        cudaError_t gpuMalloc1 = cudaMalloc((void**)&device_data_a, nBytes);
        cudaError_t gpuMalloc2 = cudaMalloc((void**)&device_data_b, nBytes);
        cudaError_t gpuMalloc3 = cudaMalloc((void**)&device_data_c, nBytes);

        if (gpuMalloc1 != cudaSuccess || gpuMalloc2 != cudaSuccess || gpuMalloc3 != cudaSuccess)
            throw std::runtime_error("Failed to allocate device memory!!!");
    }
    catch (std::exception e) {
        std::cerr << e.what();
        return -1;
    }

    // Populate data arrays
    for (int i = 0; i < N; i++) {
        host_data_a[i] =  i;
        host_data_b[i] = (i*i);
    }

    /* START TIME */
    auto start = std::chrono::high_resolution_clock::now();

    // Copy data from host-->device (cpu to gpu)
    try {
        cudaError_t memCpy1 = cudaMemcpy(device_data_a, host_data_a, nBytes, cudaMemcpyHostToDevice);
        cudaError_t memCpy2 = cudaMemcpy(device_data_b, host_data_b, nBytes, cudaMemcpyHostToDevice);

        if (memCpy1 != cudaSuccess || memCpy2 != cudaSuccess)
            throw std::runtime_error("Failed to copy host data to device!!!");
    }
    catch (std::exception e) {
        std::cerr << e.what();
        return -1;
    }
    
    // Launch kernel
    // Note: N threads per block should be multiple of 32 (= 1 warp)
    sampleAddKernel<<<1, N>>>(device_data_a, device_data_b, device_data_c);

    // Copy result back to device memory
    try {
        cudaError_t memCpy1 = cudaMemcpy(host_data_c, device_data_c, nBytes, cudaMemcpyDeviceToHost);

        if (memCpy1 != cudaSuccess)
            throw std::runtime_error("Failed to copy device data to host!!!");
    }
    catch (std::exception e) {
        std::cerr << e.what();
        return -1;
    }

    /* END TIME */
    auto end = std::chrono::high_resolution_clock::now();
    double time_taken = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    time_taken *= 1e-6;
    std::cout << "Execution time: " << std::setprecision(9) << time_taken << " ms\n";

    // Verify Results: should be i + i^2
    for (int i = 0; i < N; i++) {
        assert(host_data_c[i] == (host_data_a[i] + host_data_b[i]), "Result is incorrect!!!");
    }

    std::cout << "Vector addition complete and correct!!!\n";

    // Free memory
    free(host_data_a);
    free(host_data_b);
    free(host_data_c);
    cudaFree(device_data_a);
    cudaFree(device_data_b);
    cudaFree(device_data_c);

    std::cout << "Program finished, exiting...\n";
    return 0;
}

// Fxn that gets and prints device's warp size. Typically is 32, but still good to check...
void printDeviceWarpSize() {

    cudaDeviceProp deviceProperties;
    if (cudaGetDeviceProperties(&deviceProperties, 0) != cudaSuccess)
        std::cout << "Failed to retrieve device properties!!!\n";
    else
        std::cout << "Current device's warp size is: " << deviceProperties.warpSize << '\n';
}