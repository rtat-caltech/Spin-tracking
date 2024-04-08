#include "../include/utils.h"

void synchronize() {
#if defined(__NVCOMPILER) || defined(__NVCC__)
	gpuErrchk(cudaDeviceSynchronize());
#elif defined(__HIPCC__)
	gpuErrchk(hipDeviceSynchronize());
#endif
}

void genericFree(void* ptr) {
#if defined(__NVCOMPILER) || defined(__NVCC__)
	cudaFree(ptr);
#elif defined(__HIPCC__)
	hipFree(ptr);
#else
	free(ptr);
#endif	
}
