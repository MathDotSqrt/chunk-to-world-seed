//MathDotSqrt

//"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\bin\nvcc.exe"  -ccbin "C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Tools\MSVC\14.16.27023\bin\Hostx64\x64" -o main main.cu -O3


// IDE indexing
#ifdef __JETBRAINS_IDE__
#define __host__
#define __device__
#define __shared__
#define __constant__
#define __global__
#define __CUDACC__
#include <__clang_cuda_builtin_vars.h>
#include <__clang_cuda_cmath.h>
#include <__clang_cuda_complex_builtins.h>
#include <__clang_cuda_intrinsics.h>
#include <__clang_cuda_math_forward_declares.h>
#include <device_functions.h>
#endif

//reduce array of zeros
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <memory>
#include <array>

#include <iostream>
#include <math.h>
#include <cmath>
#include <vector>
#include <sstream>
#include <string>
#include <fstream>
#include <assert.h>

//Used for sleep function
#include <chrono>
#include <thread>

#define CHECK_GPU_ERR(code) gpuAssert((code), __FILE__, __LINE__)
inline void gpuAssert(cudaError_t code, const char* file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s (code %d) %s %d\n",
        cudaGetErrorString(code), code, file, line);
        exit(code);
    }
}

/*MAGIC NUMBERS*/
constexpr uint64_t MASK48 = ((1ULL << 48) - 1ULL);
constexpr uint64_t MASK32 = ((1ULL << 32) - 1ULL);
constexpr uint64_t MASK16 = ((1ULL << 16) - 1ULL);

constexpr uint64_t M1 = 25214903917ULL;
constexpr uint64_t ADDEND1 = 11ULL;

constexpr uint64_t M2 = 205749139540585ULL;
constexpr uint64_t ADDEND2 = 277363943098ULL;

constexpr uint64_t M4 = 55986898099985ULL;
constexpr uint64_t ADDEND4 = 49720483695876ULL;
/*MAGIC NUMBERS*/

/*DETAILS*/
constexpr int32_t MAX_LINE = 1000;
constexpr size_t OUTPUT_SEED_ARRAY_SIZE = 1 << 20;
/*DETAILS*/

/*CUDA LAUNCH CONSTANTS*/
constexpr int32_t BLOCK_SIZE = 512;
constexpr int32_t NUM_BLOCKS = 128;
constexpr int32_t NUM_WORKERS = NUM_BLOCKS * BLOCK_SIZE;
/*CUDA LAUNCH CONSTANTS*/

/*CHUNK CONSTANTS*/
// constexpr int32_t CHUNK_X = 3;
// constexpr int32_t CHUNK_Z = -3;
#define CHUNK_X 3
#define CHUNK_Z -3

/*CHUNK CONSTANTS*/

/*FILE PATHS*/
constexpr const char * INPUT_FILE_PATH = "data/chunk_seeds.txt";
constexpr const char * OUTPUT_FILE_PATH = "data/WorldSeeds.txt";
/*FILE PATHS*/

__host__ __device__
int64_t next_long(uint64_t* seed) {
  *seed = (*seed * M1 + ADDEND1) & MASK48;
  int32_t u = *seed >> 16;
  *seed = (*seed * M1 + ADDEND1) & MASK48;
  return ((uint64_t)u << 32) + (int32_t)(*seed >> 16);
}

__device__
void add_seed(uint64_t seed, uint64_t* seeds, uint64_t* seedCounter)
{
  // unsigned long long* cast is required for CUDA 9 :thonkgpu:
  uint64_t id = atomicAdd((unsigned long long*) seedCounter, 1ULL);
  seeds[id] = seed;
}

__host__ __device__
uint64_t make_mask(int32_t bits) {
  return (1ULL << bits) - 1;
}

__device__
int ctz(uint64_t v) {
  //return __popcll((v & (-v))-1);
  return __popcll(v ^ (v - 1)) - 1;
}

constexpr __host__ __device__ uint64_t mod_inv(uint64_t x) {
  uint64_t inv = 0;
  uint64_t b = 1;
  for (int32_t i = 0; i < 16; i++) {
    inv |= (1ULL << i) * (b & 1);
    b = (b - x * (b & 1)) >> 1;
  }
  return inv;
}

__host__ __device__ uint64_t get_chunk_seed(uint64_t worldSeed) {
  uint64_t seed = (worldSeed ^ M1) & MASK48;
  int64_t a = next_long(&seed) / 2 * 2 + 1;
  int64_t b = next_long(&seed) / 2 * 2 + 1;
  return (uint64_t)(((CHUNK_X * a + CHUNK_Z * b) ^ worldSeed) & MASK48);
}

__host__ __device__ uint64_t get_partial_addend(uint64_t partialSeed, int32_t bits) {
  uint64_t mask = make_mask(bits);
/* clang-format off */
  return ((uint64_t)CHUNK_X) * (((int32_t)(((M2 * ((partialSeed ^ M1) & mask) + ADDEND2) & MASK48) >> 16)) / 2 * 2 + 1) +
         ((uint64_t)CHUNK_Z) * (((int32_t)(((M4 * ((partialSeed ^ M1) & mask) + ADDEND4) & MASK48) >> 16)) / 2 * 2 + 1);
/* clang-format on */

}

__device__ void add_world_seed(uint64_t firstAddend, int32_t multTrailingZeroes, uint64_t firstMultInv,
                                    uint64_t c, uint64_t chunkSeed, uint64_t* seeds, uint64_t* seedCounter) {
  if(ctz(firstAddend) < multTrailingZeroes)
      return;
  uint64_t bottom32BitsChunkseed = chunkSeed & MASK32;

  uint64_t b = (((firstMultInv * firstAddend) >> multTrailingZeroes) ^ (M1 >> 16)) & make_mask(16 - multTrailingZeroes);
  if (multTrailingZeroes != 0) {
    uint64_t smallMask = make_mask(multTrailingZeroes);
    uint64_t smallMultInverse = smallMask & firstMultInv;
    uint64_t target = (((b ^ (bottom32BitsChunkseed >> 16)) & smallMask) -
                            (get_partial_addend((b << 16) + c, 32 - multTrailingZeroes) >> 16)) & smallMask;
    b += (((target * smallMultInverse) ^ (M1 >> (32 - multTrailingZeroes))) & smallMask) << (16 - multTrailingZeroes);
  }
  uint64_t bottom32BitsSeed = (b << 16) + c;
  uint64_t target2 = (bottom32BitsSeed ^ bottom32BitsChunkseed) >> 16;
  uint64_t secondAddend = (get_partial_addend(bottom32BitsSeed, 32) >> 16);
  secondAddend &= MASK16;
  uint64_t topBits = ((((firstMultInv * (target2 - secondAddend)) >> multTrailingZeroes) ^ (M1 >> 32)) & make_mask(16 - multTrailingZeroes));

  for (; topBits < (1ULL << 16); topBits += (1ULL << (16 - multTrailingZeroes))) {
    if (get_chunk_seed((topBits << 32) + bottom32BitsSeed) == chunkSeed) {
        add_seed((topBits << 32) + bottom32BitsSeed, seeds, seedCounter);
    }
  }

}

__global__ void crack(uint64_t seedInputCount, uint64_t* seedInputArray, uint64_t* seedOutputCounter, uint64_t* seedOutputArray,
                        int32_t multTrailingZeroes, uint64_t firstMultInv, int32_t xCount, int32_t zCount, int32_t totalCount) {
  uint64_t global_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (global_id > seedInputCount)
      return;

  uint64_t chunkSeed = seedInputArray[global_id];
  int32_t x = CHUNK_X;
  int32_t z = CHUNK_Z;

  if(CHUNK_X == 0 && CHUNK_Z == 0){
    add_seed(chunkSeed, seedOutputArray, seedOutputCounter);
  }
  else{
    uint64_t f = chunkSeed & MASK16;
    uint64_t c = xCount == zCount ? chunkSeed & ((1ULL << (xCount + 1)) - 1) :
                                    chunkSeed & ((1ULL << (totalCount + 1)) - 1) ^ (1 << totalCount);
    for (; c < (1ULL << 16); c += (1ULL << (totalCount + 1))) {
      uint64_t target = (c ^ f) & MASK16;
      uint64_t magic = (uint64_t)(x * ((M2 * ((c ^ M1) & MASK16) + ADDEND2) >> 16)) +
                       (uint64_t)(z * ((M4 * ((c ^ M1) & MASK16) + ADDEND4) >> 16));
      add_world_seed(target - (magic & MASK16), multTrailingZeroes, firstMultInv, c, chunkSeed, seedOutputArray, seedOutputCounter);
      if(CHUNK_X != 0){
        add_world_seed(target - ((magic + x) & MASK16), multTrailingZeroes, firstMultInv, c, chunkSeed, seedOutputArray, seedOutputCounter);
      }
      if(CHUNK_Z != 0 && CHUNK_X != CHUNK_Z){
        add_world_seed(target - ((magic + z) & MASK16), multTrailingZeroes, firstMultInv, c, chunkSeed, seedOutputArray, seedOutputCounter);
      }
      if(CHUNK_X != 0 && CHUNK_Z != 0 && CHUNK_X + CHUNK_Z != 0){
        add_world_seed(target - ((magic + x + z) & MASK16), multTrailingZeroes, firstMultInv, c, chunkSeed, seedOutputArray, seedOutputCounter);
      }
      if(CHUNK_X != 0 && CHUNK_X != CHUNK_Z){
        add_world_seed(target - ((magic + 2 * x) & MASK16), multTrailingZeroes, firstMultInv, c, chunkSeed, seedOutputArray, seedOutputCounter);
      }
      if(CHUNK_Z != 0 && CHUNK_X != CHUNK_Z){
        add_world_seed(target - ((magic + 2 * z) & MASK16), multTrailingZeroes, firstMultInv, c, chunkSeed, seedOutputArray, seedOutputCounter);
      }
      if(CHUNK_X != 0 && CHUNK_Z != 0 && CHUNK_X + CHUNK_Z != 0 && CHUNK_X * 2 + CHUNK_Z != 0){
        add_world_seed(target - ((magic + 2 * x + z) & MASK16), multTrailingZeroes, firstMultInv, c, chunkSeed, seedOutputArray, seedOutputCounter);
      }
      if(CHUNK_X != 0 && CHUNK_Z != 0 && CHUNK_X != CHUNK_Z && CHUNK_X + CHUNK_Z != 0 && CHUNK_X + CHUNK_Z * 2 != 0){
        //is the x supposed to be multiplied
        add_world_seed(target - ((magic + x + 2 * z) & MASK16), multTrailingZeroes, firstMultInv, c, chunkSeed, seedOutputArray, seedOutputCounter);
      }
      if(CHUNK_X != 0 && CHUNK_Z != 0 && CHUNK_X + CHUNK_Z != 0){
        add_world_seed(target - ((magic + 2 * x + 2 * z) & MASK16), multTrailingZeroes, firstMultInv, c, chunkSeed, seedOutputArray, seedOutputCounter);
      }
    }
  }
}

constexpr int32_t count_trailing_zeros(uint64_t v){
  int c = 0;
  v = (v ^ (v - 1)) >> 1;

  for(c = 0; v != 0; c++){
    v >>= 1;
  }

  return c;
}

FILE* open_file(const char *path, const char *mode){
  auto fp = fopen(path, mode);
  if(fp == nullptr){
    printf("Error: could not open file %s with mode %s", path, mode);
    exit(1);
  }
  return fp;
}

int32_t count_file_length(FILE *file){
  static char line[MAX_LINE];
  int32_t total = 0;
  while (fgets(line, MAX_LINE, file)) total++;

  //seeks to beginning of file
  rewind(file);
  return total;
}

size_t file_to_buffer(FILE *source, uint64_t *dest, size_t N){
  static char line[MAX_LINE];
  for(size_t i = 0; i < N; i++){
    if(fgets(line, MAX_LINE, source) != nullptr){
      sscanf(line, "%llu", &dest[i]);    //THIS IS SUPPOSED TO BE LLU
    }
    else{
      return i;
    }
  }
  return N;
}

void buffer_to_file(uint64_t *source, FILE *dest, size_t N){
  for(size_t i = 0; i < N; i++){
    fprintf(dest, "%llu\n", source[i]);  //THIS IS SUPPOSED TO BE LLU
  }

  fflush(dest);
}

int main(){
  setbuf(stdout, NULL);

  printf("Opening files...\n");
  FILE *in = open_file(INPUT_FILE_PATH, "r");
  FILE *out = open_file(OUTPUT_FILE_PATH, "w");

  const int32_t total_input_seeds = count_file_length(in);
  printf("Total seeds: %d\n", total_input_seeds);

  const int32_t input_seed_count = NUM_WORKERS;
  //uint64_t *input_cpu_buffer = (uint64_t*)malloc(sizeof(uint64_t) * input_seed_count);

  printf("Allocating gpu mem\n");
  uint64_t *input_seeds_cpu = (uint64_t*)malloc(sizeof(uint64_t) * input_seed_count);
  uint64_t *output_seeds_cpu = (uint64_t*)malloc(sizeof(uint64_t) * OUTPUT_SEED_ARRAY_SIZE);

  uint64_t *input_seeds_gpu = nullptr;
  uint64_t *output_seeds_gpu = nullptr;
  uint64_t *output_seed_count = nullptr;

  CHECK_GPU_ERR(cudaMalloc(&input_seeds_gpu, sizeof(uint64_t) * input_seed_count));
  CHECK_GPU_ERR(cudaMalloc(&output_seeds_gpu, sizeof(uint64_t) * OUTPUT_SEED_ARRAY_SIZE));
  CHECK_GPU_ERR(cudaMallocManaged(&output_seed_count, sizeof(uint64_t)));

  *output_seed_count = 0;

  constexpr auto first_multiplier = (M2 * (uint64_t)CHUNK_X + M4 * (uint64_t)CHUNK_Z) & MASK16;
  constexpr auto mult_trailing_zeros = count_trailing_zeros(first_multiplier);
  constexpr auto shift = first_multiplier >> mult_trailing_zeros;
  constexpr uint64_t first_mult_inv = (uint64_t)mod_inv(shift);

  constexpr auto x_count = count_trailing_zeros((uint64_t)CHUNK_X);
  constexpr auto z_count = count_trailing_zeros((uint64_t)CHUNK_Z);
  constexpr auto total_count = count_trailing_zeros(CHUNK_X | CHUNK_Z);

  //520 0 0 0
  printf("Reading input...\n");
  uint64_t file_input_count = file_to_buffer(in, input_seeds_cpu, input_seed_count);
  printf("init FILE_INPUT_COUNT %llu\n", file_input_count);

  uint64_t output_count = 0LLU;
  CHECK_GPU_ERR(cudaMemcpy(input_seeds_gpu, input_seeds_cpu, file_input_count * sizeof(uint64_t), cudaMemcpyHostToDevice));
  bool flag = false;
  while(file_input_count == input_seed_count){
    printf("Compute...\n");
    printf("INPUT %llu\n", input_seeds_cpu[0]);

    std::cout << input_seed_count << " " << mult_trailing_zeros << " " << first_mult_inv << " " << x_count << " " << z_count << " " << total_count << "\n";
    crack<<<NUM_BLOCKS, BLOCK_SIZE>>>(
      input_seed_count,
      input_seeds_gpu,
      output_seed_count,
      output_seeds_gpu,
      mult_trailing_zeros,
      first_mult_inv,
      x_count,
      z_count,
      total_count
    );

    //write output cpu to file concurrently
    printf("OUTPUT_COUNT %llu\n", output_count);
    buffer_to_file(output_seeds_cpu, out, output_count);
    //read input file to cpu concurrently
    file_input_count = file_to_buffer(in, input_seeds_cpu, input_seed_count);
    printf("FILE_INPUT_COUNT %llu\n", file_input_count);


    CHECK_GPU_ERR(cudaPeekAtLastError());
    CHECK_GPU_ERR(cudaDeviceSynchronize());

    output_count = *output_seed_count;
    printf("OUTPUT_SEED_COUNT %llu\n", output_count);
    CHECK_GPU_ERR(cudaMemcpy(input_seeds_gpu, input_seeds_cpu, file_input_count * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CHECK_GPU_ERR(cudaMemcpy(output_seeds_cpu, output_seeds_gpu, output_count * sizeof(uint64_t), cudaMemcpyDeviceToHost));
    *output_seed_count = 0;
  }

  //writing from previous run
  crack<<<NUM_BLOCKS, BLOCK_SIZE>>>(
    file_input_count,
    input_seeds_gpu,
    output_seed_count,
    output_seeds_gpu,
    mult_trailing_zeros,
    first_mult_inv,
    x_count,
    z_count,
    total_count
  );
  //write from previous run
  printf("main read %llu\n", file_input_count);
  buffer_to_file(output_seeds_cpu, out, output_count);
  CHECK_GPU_ERR(cudaPeekAtLastError());
  CHECK_GPU_ERR(cudaDeviceSynchronize());

  output_count = *output_seed_count;
  printf("OUTPUT_SEED_COUNT %llu\n", output_count);
  CHECK_GPU_ERR(cudaMemcpy(output_seeds_cpu, output_seeds_gpu, output_count * sizeof(uint64_t), cudaMemcpyDeviceToHost));
  buffer_to_file(output_seeds_cpu, out, output_count);


  printf("Done.\n");

  free(input_seeds_cpu);
  free(output_seeds_cpu);

  cudaFree(input_seeds_gpu);
  cudaFree(output_seeds_gpu);
  cudaFree(output_seed_count);
  fclose(in);
  fflush(out);
  fclose(out);
}
