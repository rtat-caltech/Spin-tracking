#include "../include/utils.h"

void synchronize() {
#if defined(__NVCOMPILER) || defined(__NVCC__)
	gpuErrchk(cudaDeviceSynchronize());
#elif defined(__HIPCC__)
	gpuErrchk(hipDeviceSynchronize());
#endif
}
