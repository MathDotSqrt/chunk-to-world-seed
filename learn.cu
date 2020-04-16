#include <iostream>

constexpr size_t N = 1 << 20;
constexpr int NUM_THREADS = 256;
constexpr int NUM_BLOCKS = (N + NUM_THREADS-1) / NUM_THREADS;

__global__
void add(size_t n, float *x, float *y){
  const int start_index = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;
  for(size_t i = start_index; i < n; i += stride){
    y[i] = x[i] + y[i];
  }
}


int main(void){
  float *x, *y;
  cudaMallocManaged(&x, N * sizeof(float));
  cudaMallocManaged(&y, N * sizeof(float));

  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }


  add<<<NUM_BLOCKS, NUM_THREADS>>>(N, x, y);
  cudaDeviceSynchronize();
  std::cout << "Done.\n";
  //std::cin.get();

  cudaFree(x);
  cudaFree(y);
}
