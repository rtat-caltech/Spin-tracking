#ifndef __RNG_H_INCLUDED__
#define __RNG_H_INCLUDED__

#include <stdint.h>
#include <iostream>
#include <math.h>

#if defined(__NVCOMPILER) || defined(__NVCC__)
#define __PREPROC__ __host__ __device__
#include <cuda_runtime.h>
#elif defined(__HIPCC__)
#define __PREPROC__ __host__ __device__
#include <hip/hip_runtime.h>
#else
#define __PREPROC__
#endif

#include "double3.h"

struct rngState{
	uint64_t x;
	uint64_t y;
	uint64_t z;
	uint64_t w;
    _PREC spare; //used for the normal generator
    bool hasSpare = false; //used for the normal generator
};
__PREPROC__ static inline double DoubleFromBits(const uint64_t i) {
	return (i >> 11) * 0x1.0p-53;
};
__PREPROC__ void initialize_xoshiro_state(rngState& state, uint64_t seed);
__PREPROC__ uint64_t rol64(const uint64_t x, const int k);
__PREPROC__ uint64_t splitmix64(uint64_t& state);
__PREPROC__ uint64_t xoshiro256p(rngState& state);
__PREPROC__ _PREC uniform(rngState& state);
__PREPROC__ _PREC uniform(rngState& state, const _PREC low, const _PREC high);
__PREPROC__ _PREC normal(rngState& state, const _PREC mean, const _PREC std);
__PREPROC__ _PREC maxboltz(rngState& state, const _PREC sqrtkT_m);
__PREPROC__ _PREC unif02pi(rngState& state);
__PREPROC__ _PREC exponential(rngState& state, const _PREC tc);
#endif
