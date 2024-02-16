#ifndef __SIMULATION_H_DEFINED__
#define __SIMULATION_H_DEFINED__

#include <stdio.h>
#include <cmath>
#include <math.h>
#include <cctype>
#include <iostream>

#include "../include/double3.h"
#include "../include/options.h"
#include "../include/coeff.h"
#include "../include/particle.h"
#include "../include/outputHandling.h"

#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
#include <hiprand/hiprand.h>
#include <hiprand/hiprand_kernel.h>
#elif defined(__NVCOMPILER) || defined(__NVCC__)
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#else
#include <random>
#define __PREPROC__
#endif

void mainAnalysis(const options opt, int totalTime, char* outputName, unsigned int seed);

#endif
