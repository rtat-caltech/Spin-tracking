#include "../include/simulation.h"
#include <unistd.h>
#include <chrono>

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

outputBuffers createOutputBuffers(options opt){
	outputBuffers buffers;
	if(tolower(opt.output=='a') || tolower(opt.output=='n') || tolower(opt.output=='h')){
		unsigned int numOutput = int(ceil(double(opt.tf - opt.t0)/opt.ioutInt));
		buffers.particleStatesCPU = (outputDtype*)malloc(sizeof(outputDtype)*opt.numParticles);
		#if defined(__HIPCC__)
		gpuErrchk(hipMalloc(&buffers.particleStatesGPU, sizeof(outputDtype)*opt.numParticles));
		#elif defined(__NVCC__) || defined(__NVCOMPILER)
		gpuErrchk(cudaMalloc(&buffers.particleStatesGPU, sizeof(outputDtype)*opt.numParticles));
		#endif
		if(tolower(opt.output) == 'a'){
			//This means output the average and standard deviation of the particle data at the different output interval times
			buffers.times = (double*)malloc(sizeof(double)*numOutput);
			buffers.averagesCPU = (double*)malloc(sizeof(double)*numOutput);
			buffers.stdsCPU = (double*)malloc(sizeof(double)*numOutput);
			buffers.temp = (double*)malloc(sizeof(double) * opt.numParticles);
		}
		else if(tolower(opt.output) == 'n'){
			//This is the standard dump all particle information method
			buffers.allParticleStatesCPU = (outputDtype*)malloc(sizeof(outputDtype)*numOutput*opt.numParticles);
		}
		else if(tolower(opt.output) == 'h'){
			buffers.numPhiBins = opt.numPhiBins;
			buffers.numThetaBins = opt.numThetaBins;
			buffers.numx = opt.posHistBins.x;
			buffers.numy = opt.posHistBins.y;
			buffers.numz = opt.posHistBins.z;
			buffers.posHist = (unsigned int*)malloc(sizeof(unsigned int) * buffers.numx * buffers.numy * buffers.numz);
			buffers.thetaHist = (unsigned int*)malloc(sizeof(unsigned int) * buffers.numThetaBins);
			buffers.phiHist = (unsigned int*)malloc(sizeof(unsigned int) * buffers.numPhiBins);
		}
		return buffers;
	}
	else{
		printf("unrecognized input option\n");
		printf("code exiting\n");
		exit(-1);
		return buffers;
	}
	return buffers;
}

void destroyOutputBuffers(outputBuffers buffers, options opt){
	#if defined(__HIPCC__)
	gpuErrchk(hipFree(buffers.particleStatesGPU));
	#elif defined(__NVCC__) || defined(__NVCOMPILER)
	gpuErrchk(cudaFree(buffers.particleStatesGPU));
	#endif
	if(opt.output == 'A' || opt.output == 'a'){
		free(buffers.times);
		free(buffers.averagesCPU);
		free(buffers.stdsCPU);
		free(buffers.temp);
	}
	else if(opt.output == 'N' || opt.output == 'n'){
		free(buffers.allParticleStatesCPU);
	}
	else if(opt.output == 'H' || opt.output == 'h'){
		free(buffers.posHist);
		free(buffers.thetaHist);
		free(buffers.phiHist);
	}
	return;
}

#if defined(__HIPCC__) || defined(__NVCOMPILER) || defined(__NVCC__)
__global__ void initializeParticles(particle* particles, int numParticles, options OPT, outputBuffers buffers, unsigned long seed){
	unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid < numParticles){
		particles[tid] = particle(OPT.yi, OPT, seed, tid); //save the particle to the array
		buffers.particleStatesGPU[tid] = particles[tid].getState(); //save the state of the particle for the CPU to handle the output
		//printf("%d \n", tid);
	}
}

__global__ void runSimulation(particle* particles, int numParticles, options OPT, outputBuffers buffers, double nextTOut){
	unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid < numParticles){
		particles[tid].updateTF(nextTOut);
		particles[tid].run();
		buffers.particleStatesGPU[tid] = particles[tid].getState(); //save the state of the particle for the CPU to handle the output
	}
}

#else
void initializeParticles(particle* particles, int numParticles, options OPT, outputBuffers buffers, unsigned long seed){
	#if defined(_OPENMP)
	#pragma omp parallel for
	#endif
	for(unsigned int tid = 0; tid < numParticles; tid++){
		particles[tid] = particle(OPT.yi, OPT, seed, tid);
		buffers.particleStatesCPU[tid] = particles[tid].getState();
	}
}

void runSimulation(particle* particles, int numParticles, options OPT, outputBuffers buffers, double nextTOut){
	#if defined(_OPENMP)
	#pragma omp parallel for
	#endif
	for(unsigned int tid = 0; tid < numParticles; tid++){
		particles[tid].updateTF(nextTOut);
		particles[tid].run();
		buffers.particleStatesCPU[tid] = particles[tid].getState(); 
	}
}
#endif

void calculateMeanAndSD(double* data, int length, double &average, double &std) {
	average = 0.0;
	std = 0.0;
	double sum = 0.0;
	int i;
	for(i = 0; i < length; ++i) {
		sum += data[i];
	}
	average = sum / double(length);
	for(i = 0; i < length; ++i) {
		std += (data[i]-average)*(data[i]-average);
	};
	std = sqrt(std / double(length));
	return;
}

void histogramPos(int length, outputBuffers &buffers, options opt){
	for(int i = 0; i < buffers.numx*buffers.numy*buffers.numz; i++){
		buffers.posHist[i] = 0; //reset the histogram to zero
	}
	double3 bins;
	int bin;
	for(int i = 0; i < length; i++){
		bins = (buffers.particleStatesCPU[i].x+opt.L/2.0)/opt.L*double3{(double)buffers.numx, (double)buffers.numy, (double)buffers.numz};
		bin = (int)bins.x + (int)bins.y*buffers.numx + (int)bins.z*buffers.numx*buffers.numy;
		buffers.posHist[bin] += 1;
	}
}

void histogramSpin(int length, outputBuffers &buffers, options opt){
	for(int i = 0; i < buffers.numPhiBins; i++){
		buffers.phiHist[i] = 0; //reset the histogram to zero
	}
	for(int i = 0; i < buffers.numThetaBins; i++){
		buffers.thetaHist[i] = 0;
	}
	int bin;
	double theta, phi, lengthVec;
	for(int i = 0; i < length; i++){
		//phi is in the x-y plane from -pi to pi, theta is for the aximuthal angle from -pi/2 to pi/2
		phi = atan2(buffers.particleStatesCPU[i].s.y, buffers.particleStatesCPU[i].s.x);
		bin = (phi+M_PI)/(2.0*M_PI)*buffers.numPhiBins;
		if(bin < 0)
			bin = 0;
		else if(bin > buffers.numPhiBins-1)
			bin = buffers.numPhiBins-1;
		buffers.phiHist[bin]+=1;
		
		lengthVec = len(buffers.particleStatesCPU[i].s);
		theta = acos(buffers.particleStatesCPU[i].s.z/length);
		bin = theta/M_PI * buffers.numThetaBins;
		if(bin < 0)
			bin = 0;
		else if(bin > buffers.numThetaBins-1)
			bin = buffers.numThetaBins-1;
		buffers.thetaHist[bin]+=1;
	}
}

void handleOutput(FILE * f, particle* particles, options opt, outputBuffers buffers){
	if(tolower(opt.output) == 'a'){
		//in this case we need to move the data back to the CPU and then calculate the average and standard deviation
		#if defined(__NVCOMPILER) || defined(__NVCC__)
		gpuErrchk(cudaMemcpy(buffers.particleStatesCPU, buffers.particleStatesGPU, sizeof(outputDtype)*opt.numParticles, cudaMemcpyDeviceToHost));
		gpuErrchk(cudaDeviceSynchronize());
		#elif defined(__HIPCC__)
		gpuErrchk(hipMemcpy(buffers.particleStatesCPU, buffers.particleStatesGPU, sizeof(outputDtype)*opt.numParticles, hipMemcpyDeviceToHost));
		gpuErrchk(hipDeviceSynchronize());
		#endif
		//first save the current time we are at
		fwrite(&buffers.particleStatesCPU[0].t, sizeof(double), 1, f);
		double average, std;
		//x component of spin
		for(int i = 0; i < opt.numParticles; i++){
			buffers.temp[i] = buffers.particleStatesCPU[i].s.x;
		}
		calculateMeanAndSD(buffers.temp, opt.numParticles, average, std);
		fwrite(&average, sizeof(double), 1, f);
		fwrite(&std, sizeof(double), 1, f);
		//y component of spin
		for(int i = 0; i < opt.numParticles; i++){
			buffers.temp[i] = buffers.particleStatesCPU[i].s.y;
		}
		calculateMeanAndSD(buffers.temp, opt.numParticles, average, std);
		fwrite(&average, sizeof(double), 1, f);
		fwrite(&std, sizeof(double), 1, f);
		//z component of spin
		for(int i = 0; i < opt.numParticles; i++){
			buffers.temp[i] = buffers.particleStatesCPU[i].s.z;
		}
		calculateMeanAndSD(buffers.temp, opt.numParticles, average, std);
		fwrite(&average, sizeof(double), 1, f);
		fwrite(&std, sizeof(double), 1, f);
	}
	else if(tolower(opt.output) == 'n'){
		//in this case we're doing the bulk output of all particle data
		#if defined(__NVCOMPILER) || defined(__NVCC__)
		gpuErrchk(cudaMemcpy(buffers.particleStatesCPU, buffers.particleStatesGPU, sizeof(outputDtype)*opt.numParticles, cudaMemcpyDeviceToHost));
		gpuErrchk(cudaDeviceSynchronize());
		#elif defined(__HIPCC__)
		gpuErrchk(hipMemcpy(buffers.particleStatesCPU, buffers.particleStatesGPU, sizeof(outputDtype)*opt.numParticles, hipMemcpyDeviceToHost));
		gpuErrchk(hipDeviceSynchronize());
		#endif
		fwrite(buffers.particleStatesCPU, sizeof(outputDtype) * opt.numParticles, 1, f);
	}
	else if(tolower(opt.output) == 'h'){
		//in this case we're doing the histogramming of the data
		#if defined(__NVCOMPILER) || defined(__NVCC__)
		gpuErrchk(cudaMemcpy(buffers.particleStatesCPU, buffers.particleStatesGPU, sizeof(outputDtype)*opt.numParticles, cudaMemcpyDeviceToHost));
		gpuErrchk(cudaDeviceSynchronize());
		#elif defined(__HIPCC__)
		gpuErrchk(hipMemcpy(buffers.particleStatesCPU, buffers.particleStatesGPU, sizeof(outputDtype)*opt.numParticles, hipMemcpyDeviceToHost));
		gpuErrchk(hipDeviceSynchronize());
		#endif
		//write what time it currently is
		fwrite(&buffers.particleStatesCPU[0].t, sizeof(double), 1, f);
		histogramPos(opt.numParticles, buffers, opt);
		fwrite(buffers.posHist, sizeof(unsigned int), buffers.numx*buffers.numy*buffers.numz, f);
		histogramSpin(opt.numParticles, buffers, opt);
		fwrite(buffers.phiHist, sizeof(unsigned int), buffers.numPhiBins, f);
		fwrite(buffers.thetaHist, sizeof(unsigned int), buffers.numThetaBins, f);
	}
}

//this functions does the actual analysis and integration
void mainAnalysis(options opt, int totalTime, char* outputName, unsigned int seed){
	#if defined(__NVCOMPILER) || defined(__HIPCC__) || defined(__NVCC__)
	{
		//In this case we're going to use the GPU to do mostly everything
		//start up the same way basically
		//start the clock on the process
		//Do the single CPU version of the code that uses all cores/threads on a singular CPU
		unsigned int timestamp = time(NULL);
		outputBuffers buffers = createOutputBuffers(opt);
		
		//now actually do the kernel call
		int numPartsPerBlock = opt.numPerGPUBlock;
		int numBlocks = std::ceil((double)opt.numParticles/(double)numPartsPerBlock);
		
		particle* particles;
		#if defined(__NVCOMPILER) || defined(__NVCC__)
		gpuErrchk(cudaMalloc(&particles, sizeof(particle) * opt.numParticles));
		#elif defined(__HIPCC__)
		gpuErrchk(hipMalloc(&particles, sizeof(particle) * opt.numParticles));
		#endif
		//create the output file
		FILE* f = fopen(outputName, "wb");
		fwrite(&opt, sizeof(options), 1, f);//write the options that were used to create the simulation
		//now initialize all of the particles in the system
        printf("initializing\n");
		initializeParticles<<<numBlocks, numPartsPerBlock>>>(particles, opt.numParticles, opt, buffers, seed);
        printf("initialized\n");
		handleOutput(f, particles, opt, buffers); //save the initial states
		unsigned int numIterations = int(floor(double(opt.tf - opt.t0)/opt.ioutInt));
		
		auto start = std::chrono::high_resolution_clock::now();
		auto stop = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop-start).count();
		for(int i = 0; i < numIterations; i++){
			double nextTime = ((double)i+1.0)*opt.ioutInt; //figure out the next stop time for the particles
			start = std::chrono::high_resolution_clock::now();
			runSimulation<<<numBlocks, numPartsPerBlock>>>(particles, opt.numParticles, opt, buffers, nextTime);
			handleOutput(f, particles, opt, buffers);
			stop = std::chrono::high_resolution_clock::now();
        		auto duration = std::chrono:: duration_cast<std::chrono::milliseconds>(stop-start).count();
			std::cout<<i<<", "<<nextTime<<", "<<duration<<std::endl;
		}
		fclose(f);
		destroyOutputBuffers(buffers, opt);
	}
	#else
	{
		//Do the single CPU version of the code that uses all cores/threads on a singular CPU
		unsigned int timestamp = time(NULL);
		outputBuffers buffers = createOutputBuffers(opt);
		//allocate the various particles
		particle* particles = (particle*)malloc(sizeof(particle) * opt.numParticles);
		//create the output file
		FILE* f = fopen(outputName, "wb");
		fwrite(&opt, sizeof(options), 1, f);//write the options that were used to create the simulation
		//initialize the particles and save their states
        printf("initializing\n");
		initializeParticles(particles, opt.numParticles, opt, buffers, seed);
        printf("initialized\n");
		handleOutput(f, particles, opt, buffers); //save the initial states
		
		unsigned int numIterations = int(floor(double(opt.tf - opt.t0)/opt.ioutInt));
        auto start = std::chrono::high_resolution_clock::now();
		auto stop = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop-start).count();
		for(int i = 0; i < numIterations; i++){
			double nextTime = ((double)i+1.0)*opt.ioutInt; //figure out the next stop time for the particles
            start = std::chrono::high_resolution_clock::now();
			runSimulation(particles, opt.numParticles, opt, buffers, nextTime);
			handleOutput(f, particles, opt, buffers);
            stop = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono:: duration_cast<std::chrono::milliseconds>(stop-start).count();
            std::cout<<i<<", "<<nextTime<<", "<<duration<<std::endl;
		}
		fclose(f);
		
		destroyOutputBuffers(buffers, opt);
	}
	#endif
	return;
}
