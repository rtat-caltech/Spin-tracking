#include "../include/simulation.h"
#include <unistd.h>
#include <chrono>
#include <iostream>
#include <fstream>

//this functions does the actual analysis and integration
void mainAnalysis(options opt, int totalTime, const char* outputName, unsigned int seed){
	_PREC k = 1.380649e-23;
	opt.tc = 1.6e-4*opt.m/(k*pow(opt.T, 8.0));
	opt.sqrtKT_m = sqrt(k*opt.T/opt.m);
	
	#if defined(__NVCOMPILER) || defined(__HIPCC__) || defined(__NVCC__)
	{
		//In this case we're going to use the GPU to do mostly everything
		//start up the same way basically
		//start the clock on the process
		//Do the single CPU version of the code that uses all cores/threads on a singular CPU
		unsigned int timestamp = time(NULL);
		//outputBuffers buffers = createOutputBuffers(opt);
		
		//now actually do the kernel call
		int numPartsPerBlock = opt.numPerGPUBlock;
		int numBlocks = std::ceil((_PREC)opt.numParticles/(_PREC)numPartsPerBlock);
				
		//now initialize all of the particles in the system
        particle p(opt);
        #if defined(__NVCOMPILER) || defined(__NVCC__)
        gpuErrchk(cudaDeviceSynchronize());
        #elif defined(__HIPCC__)
        gpuErrchk(hipDeviceSynchronize());
        #endif
        p.initParticles();
        #if defined(__NVCOMPILER) || defined(__NVCC__)
        gpuErrchk(cudaDeviceSynchronize());
        #elif defined(__HIPCC__)
        gpuErrchk(hipDeviceSynchronize());
        #endif
		//create the output file
		//this automatically writes parameters to file
		OutputHandler oh(opt, outputName);		
		p.outputData(oh); //save the initial states
		unsigned int numIterations = int(floor(_PREC(opt.tf - opt.t0)/opt.ioutInt));
		
		auto start = std::chrono::high_resolution_clock::now();
		auto stop = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop-start).count();
		for(int i = 0; i < numIterations; i++){
			_PREC nextTime = ((_PREC)i+1.0)*opt.ioutInt; //figure out the next stop time for the particles
			start = std::chrono::high_resolution_clock::now();
			p.runSimulation(nextTime);
            #if defined(__NVCOMPILER) || defined(__NVCC__)
            gpuErrchk(cudaDeviceSynchronize());
            #elif defined(__HIPCC__)
            gpuErrchk(hipDeviceSynchronize());
            #endif
			p.outputData(oh);
			stop = std::chrono::high_resolution_clock::now();
			auto duration = std::chrono:: duration_cast<std::chrono::milliseconds>(stop-start).count();
			std::cout<<"iter "<<i<<", duration "<<nextTime<<", "<<duration<<std::endl;
		}
		p.postProcess(oh);
		oh.close();		
	}
	#else
	{
		//Do the single CPU version of the code that uses all cores/threads on a singular CPU
		//In this case we're going to use the GPU to do mostly everything
		//start up the same way basically
		//start the clock on the process
		//Do the single CPU version of the code that uses all cores/threads on a singular CPU
		unsigned int timestamp = time(NULL);
		//outputBuffers buffers = createOutputBuffers(opt);
		//create the output file
		OutputHandler oh(opt, outputName);
		//now initialize all of the particles in the system
        particle p(opt);
        p.initParticles();
		p.outputData(oh); //save the initial states
		unsigned int numIterations = int(floor(_PREC(opt.tf - opt.t0)/opt.ioutInt));
		
		auto start = std::chrono::high_resolution_clock::now();
		auto stop = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop-start).count();
		for(int i = 0; i < numIterations; i++){
			_PREC nextTime = ((_PREC)i+1.0)*opt.ioutInt; //figure out the next stop time for the particles
			start = std::chrono::high_resolution_clock::now();
			p.runSimulation(nextTime);
			p.outputData(oh);
			stop = std::chrono::high_resolution_clock::now();
			auto duration = std::chrono:: duration_cast<std::chrono::milliseconds>(stop-start).count();
			std::cout<<"iter "<<i<<", duration "<<nextTime<<", "<<duration<<std::endl;
		}
		p.postProcess(oh);
		oh.close();
	}
	#endif
	return;
}

bool pathExists(hid_t id, const std::string& path) {
	return H5Lexists( id, path.c_str(), H5P_DEFAULT ) > 0;
}
