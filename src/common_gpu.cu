#include <chrono>
#include <iostream>

#include "common_gpu.cuh"
using namespace std;
using namespace GPU;

using namespace std::chrono;

__global__ void kernelAddVectors(float* vec1, float* vec2, float* vec_out,
                                 size_t n_elements) {
  int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (tid < n_elements) {
    vec_out[tid] = vec1[tid] + vec2[tid];
  }
}

void Common::addVectorsH2H(float* vec1, float* vec2, float* vec3,
                           const size_t n_elemets) {
  auto start = high_resolution_clock::now();
  auto end = high_resolution_clock::now();
  auto duration = duration_cast<microseconds>(end - start);
  
  size_t bytes = sizeof(float) * n_elemets;
  
  start = high_resolution_clock::now();
  cudaMemcpy(gpu_vec1_, vec1, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_vec2_, vec2, bytes, cudaMemcpyHostToDevice);
  end = high_resolution_clock::now();
  duration = duration_cast<microseconds>(end - start);
  cout << "H2D time: " << duration.count() << "\n";

  start = high_resolution_clock::now();
  
  auto NUM_BLOCKS = static_cast<unsigned int>(
      (n_elemets + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);

  
  kernelAddVectors<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(gpu_vec1_, gpu_vec2_,
                                                      gpu_vec3_, n_elemets);
  end = high_resolution_clock::now();
  duration = duration_cast<microseconds>(end - start);
  cout << "kernel time: " << duration.count() << "\n";

  start = high_resolution_clock::now();
  cudaMemcpy(vec3, gpu_vec3_, bytes, cudaMemcpyDeviceToHost);
  end = high_resolution_clock::now();
  duration = duration_cast<microseconds>(end - start);
  cout << "D2H time: " << duration.count() << "\n";
}

Common::Common(size_t n_elements) {
  size_t bytes = sizeof(float) * n_elements;
  cudaMalloc(&gpu_vec1_, bytes);
  cudaMalloc(&gpu_vec2_, bytes);
  cudaMalloc(&gpu_vec3_, bytes);
}

void Common::test(size_t n_elements) {
  size_t bytes = sizeof(float) * n_elements;
  cudaMalloc(&gpu_vec1_, bytes);
  cudaMalloc(&gpu_vec2_, bytes);
  cudaMalloc(&gpu_vec3_, bytes);

  cudaFree(gpu_vec1_);
  cudaFree(gpu_vec2_);
  cudaFree(gpu_vec3_);
}