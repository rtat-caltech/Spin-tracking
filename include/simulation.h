#ifndef __SIMULATION_H_DEFINED__
#define __SIMULATION_H_DEFINED__

#include <stdio.h>
#include <cmath>
#include <math.h>
#include <cctype>
#include <iostream>

#include "../include/double3.h"
#include "../include/utils.h"
#include "../include/quaternion.h"
#include "../include/options.h"
#include "../include/coeff.h"
#include "../include/logger.h"
#include "../include/particle.h"
#include "../include/floquet.h"

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

struct outputBuffers{
	outputDtype* particleStatesGPU; //where to store all the particle states on GPU
	outputDtype* particleStatesCPU; //where to store all particle states on CPU
	double *times;
	
	//information for the averages
	//right now assume this is done on the CPU
	double* averagesGPU;
	double* averagesCPU;
	double* stdsGPU;
	double* stdsCPU;
	
	//information for the standard dump method
	outputDtype* allParticleStatesGPU;
	outputDtype* allParticleStatesCPU;
	
	//information for the histogram method
	//right now assume this processing is done on the CPU
	double gridSize;
	double vecBinSize;
	int numx, numy, numz;
	int numPhiBins, numThetaBins;
	unsigned int* posHist;
	unsigned int* thetaHist;
	unsigned int* phiHist;
	
	double* temp;
};


void createOutputBuffers(options opt, void** buffers);
void mainAnalysis(const options opt, int totalTime, const char* outputName, unsigned int seed);

#if defined(__HIPCC__)
__global__ void runSimulation(particle * particles, outputBuffers* buffers, options OPT);
__global__ void initializeParticles(particle * particles, int numParticles, options OPT, unsigned long seed, coords yi);
#elif defined(__NVCOMPILER) || defined(__NVCC__)
__global__ void runSimulation(particle * particles, outputBuffers* buffers, options OPT);
__global__ void initializeParticles(particle * particles, int numParticles, options OPT, unsigned long seed, coords yi);
#else
void runSimulation(particle * particles, outputBuffers* buffers, options OPT);
void initializeParticles(particle * particles, int numParticles, options OPT, unsigned long seed, coords yi);

void aggregateSpectrum(particle*, CovarianceSpectrum&, int);

#endif

#endif
