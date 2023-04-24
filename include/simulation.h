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
	int numVecBins;
	unsigned int* xHist;
	unsigned int* yHist;
	unsigned int* zHist;
	
	unsigned int* sxHist;
	unsigned int* syHist;
	unsigned int* szHist;
	
	double* temp;
};


void createOutputBuffers(options opt, void** buffers);
void mainAnalysis(options opt, int totalTime, char* outputName, unsigned int seed);

#if defined(__HIPCC__)
__global__ void runSimulation(particle * particles, outputBuffers* buffers, options OPT);
__global__ void initializeParticles(particle * particles, int numParticles, options OPT, unsigned long seed, double3 yi);
#elif defined(__NVCOMPILER) || defined(__NVCC__)
__global__ void runSimulation(particle * particles, outputBuffers* buffers, options OPT);
__global__ void initializeParticles(particle * particles, int numParticles, options OPT, unsigned long seed, double3 yi);
#else
void runSimulation(particle * particles, outputBuffers* buffers, options OPT);
void initializeParticles(particle * particles, int numParticles, options OPT, unsigned long seed, double3 yi);
#endif

#endif
