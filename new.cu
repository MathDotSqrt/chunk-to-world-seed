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


#include <iostream>
#include <math.h>
#include <cmath>
#include <vector>
#include <sstream>
#include <string>
#include <fstream>

//Used for sleep function
#include <chrono>
#include <thread>

#define CHECK_GPU_ERR(code) gpuAssert((code), __FILE__, __LINE__)
inline void gpuAssert(cudaError_t code, const char* file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s (code %d) %s %d\n", cudaGetErrorString(code), code, file, line);
        exit(code);
    }
}

/*MAGIC NUMBERS*/
constexpr uint64_t MASK48 = ((1ULL << 48) - 1ULL);
constexpr uint64_t MASK32 = ((1ULL << 32) - 1ULL);
constexpr uint64_t MASK16 = ((1ULL << 16) - 1ULL);
constexpr uint64_t M1 = 25214903917ULL;
constexpr uint64_t APPEND1 = 11ULL;

constexpr uint64_t M2 = 205749139540585ULL;
constexpr uint64_t ADDEND2 = 277363943098ULL;

constexpr uint64_t M4 = 55986898099985ULL;
constexpr uint64_t APPEND4 = 49720483695876ULL;
/*MAGIC NUMBERS*/


constexpr int BLOCK_SIZE = 256;
constexpr int NUM_BLOCKS = 256
constexpr int NUM_WORKERS = NUM_BLOCKS * BLOCK_SIZE;





constexpr int CHUNK_X = 3;
constexpr int CHUNK_Z = -3;

constexpr int MAX_LINE = 1000;
