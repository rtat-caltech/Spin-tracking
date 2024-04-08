#ifndef __UTILS_H_DEFINED__
#define __UTILS_H_DEFINED__

#include <iostream>
#include <stdint.h>

#if defined(_OPENMP)
#include <omp.h>
#endif
#if defined(__HIPCC__)
#define __PREPROCD__ __device__
#define __PREPROC__ __host__ __device__
#elif defined(__NVCOMPILER) || defined(__NVCC__)
#define __PREPROCD__ __device__
#define __PREPROC__ __host__ __device__
#include <cuda_runtime.h>
#else
#define __PREPROCD__
#define __PREPROC__
#endif

#if defined(__NVCC__) || defined(__NVCOMPILER)
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
#elif defined(__HIPCC__)
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(hipError_t code, const char *file, int line, bool abort=true)
{
   if (code != hipSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", hipGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
#endif

void synchronize();
void genericFree(void* ptr);

template <typename T>
void genericMalloc(T** ptr, int n) {
#if defined(__NVCOMPILER) || defined(__NVCC__)
	gpuErrchk(cudaMallocManaged(ptr, sizeof(T) * n));
#elif defined(__HIPCC__)
	hipMallocManaged(ptr, sizeof(T) * n);
#else
	*ptr = (T*) malloc(sizeof(T) * n);
#endif	
}
#endif
