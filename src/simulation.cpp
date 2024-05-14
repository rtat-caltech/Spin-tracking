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
   
	//Do the single CPU version of the code that uses all cores/threads on a singular CPU
	unsigned int timestamp = time(NULL);
	//outputBuffers buffers = createOutputBuffers(opt);
						
	//now initialize all of the particles in the system
	particle p(opt);
	p.initParticles();
	//create the output file
	//this automatically writes parameters to file
	std::unique_ptr<Logger> log = createLogger(opt, outputName);
	p.outputData(log.get()); //save the initial states
	unsigned int numIterations = int(floor(_PREC(opt.tf - opt.t0)/opt.ioutInt));

	//start the clock on the process
	auto start = std::chrono::high_resolution_clock::now();
	auto stop = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop-start).count();
	for(int i = 0; i < numIterations; i++){
		_PREC nextTime = ((_PREC)i+1.0)*opt.ioutInt; //figure out the next stop time for the particles
		start = std::chrono::high_resolution_clock::now();
		p.runSimulation(nextTime);
		p.outputData(log.get());
		stop = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono:: duration_cast<std::chrono::milliseconds>(stop-start).count();
		std::cout<<"iter "<<i<<", duration "<<nextTime<<", "<<duration<<std::endl;
	}
	stop = std::chrono::high_resolution_clock::now();
	duration = std::chrono:: duration_cast<std::chrono::milliseconds>(stop-start).count();
	p.postProcess(log.get(), duration);
	return;
}
