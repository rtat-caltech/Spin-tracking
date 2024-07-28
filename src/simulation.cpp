#include "../include/simulation.h"
#include <unistd.h>
#include <chrono>
#include <iostream>
#include <fstream>

//this functions does the actual analysis and integration
void mainAnalysis(options opt, int totalTime, const char* outputName, unsigned int seed){   
	//Do the single CPU version of the code that uses all cores/threads on a singular CPU
	unsigned int timestamp = time(NULL);
	//outputBuffers buffers = createOutputBuffers(opt);
						
	//now initialize all of the particles in the system
	particle p(opt);
	p.initParticles();
	//create the output file
	//this automatically writes parameters to file
	int n_records = isFloquet(opt) ? 1 : opt.numParticles;
	std::unique_ptr<Logger> log = createLogger(opt, outputName, n_records);

	//start the clock on the process
	auto start = std::chrono::high_resolution_clock::now();
	auto stop = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop-start).count();
	RangeUnion stopTimes;
	if (isFloquet(opt)) {
		stopTimes.push_back(Range(opt.t0, opt.tf, opt.ioutInt));
	} else {
		stopTimes.concatenate(opt.stopTimes);
	}
	int i = 0;
	while (stopTimes.hasNext()) {
		_PREC nextTime = stopTimes.next();
		p.runSimulation(nextTime);
		p.outputData(log.get());
        auto tnow = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono:: duration_cast<std::chrono::milliseconds>(tnow - stop).count();
        stop = tnow;
		std::cout<<"iter "<<i<<", duration "<<nextTime<<", "<<duration<<std::endl;
		i += 1;
	}
	// unsigned int numIterations = int(floor(_PREC(opt.tf - opt.t0)/opt.ioutInt));
	// for(int i = 0; i < numIterations; i++){
	// 	_PREC nextTime = ((_PREC)i+1.0)*opt.ioutInt; //figure out the next stop time for the particles
	// 	p.runSimulation(nextTime);
	// 	p.outputData(log.get());
    //     auto tnow = std::chrono::high_resolution_clock::now();
	// 	auto duration = std::chrono:: duration_cast<std::chrono::milliseconds>(tnow - stop).count();
    //     stop = tnow;
	// 	std::cout<<"iter "<<i<<", duration "<<nextTime<<", "<<duration<<std::endl;
	// }
	stop = std::chrono::high_resolution_clock::now();
	duration = std::chrono:: duration_cast<std::chrono::milliseconds>(stop-start).count();
	p.postProcess(log.get(), duration);
	return;
}
