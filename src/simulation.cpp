#include "../include/simulation.h"
#include <unistd.h>
#include <chrono>

//this functions does the actual analysis and integration
void mainAnalysis(options opt, int totalTime, char* outputName, unsigned int seed){
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
		
		//create the output file
		FILE* f = fopen(outputName, "wb");
		fwrite(&opt, sizeof(options), 1, f);//write the options that were used to create the simulation
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
		p.outputData(f); //save the initial states
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
			p.outputData(f);
			stop = std::chrono::high_resolution_clock::now();
            		auto duration = std::chrono:: duration_cast<std::chrono::milliseconds>(stop-start).count();
			std::cout<<"iter "<<i<<", duration "<<nextTime<<", "<<duration<<std::endl;
		}
		fclose(f);
		//destroyOutputBuffers(buffers, opt);
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
		FILE* f = fopen(outputName, "wb");
		fwrite(&opt, sizeof(options), 1, f);//write the options that were used to create the simulation
		//now initialize all of the particles in the system
		particle p(opt);
		p.initParticles();
		CovarianceSpectrum cspec;
		floquetDiagonalization fd = p.initializeSpectra(cspec, opt);
		p.outputData(f); //save the initial states
		unsigned int numIterations = int(floor(_PREC(opt.tf - opt.t0)/opt.ioutInt));
		
		auto start = std::chrono::high_resolution_clock::now();
		auto stop = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop-start).count();
		for(int i = 0; i < numIterations; i++){
                        _PREC nextTime = ((_PREC)i+1.0)*opt.ioutInt; //figure out the next stop time for the particles
			start = std::chrono::high_resolution_clock::now();
			p.runSimulation(nextTime);
			if (opt.integratorType == 3) {
				int n_samp = first_sample_point(((double) i)*opt.ioutInt, opt.h) - first_sample_point(nextTime, opt.h);
				p.aggregateSpectrum(cspec, opt.numParticles);				
} else {
				p.outputData(f);
			}
			
			stop = std::chrono::high_resolution_clock::now();
            		auto duration = std::chrono:: duration_cast<std::chrono::milliseconds>(stop-start).count();
			std::cout<<"iter "<<i<<", duration "<<nextTime<<", "<<duration<<std::endl;
		}
		cspec.normalize();
		double Delta[2][2][NK] = {{{0}}};
		double X[2][2][NK] = {{{0}}};
		double Gamma[2][2][NK] = {{{0}}};
		double A[2][2] = {{0}};

		vector<pair<quaternion, Spectrum>> specs = cspec.extract();
		Matrix2cd rho = bloch_to_density(opt.yi, fd.f_modes_0);
		cout << rho << endl;
		for (int i = 0; i < specs.size(); i++) {
			quaternion c_op = specs.at(i).first;
			Spectrum spec = specs.at(i).second;
			floquet_master_equation_rates(fd, c_op, 2*M_PI/opt.w, spec, Delta, X, Gamma, A);
		}
		cout << density_to_bloch(bloch_to_density(opt.yi)) << endl;
		cout << fd.propagators[fd.n_prop - 1] << endl;
		rho = integrateFloquetMarkov(opt.t0, opt.tf, rho, A);
		int n_period = round((opt.tf - opt.t0) * opt.w/(2 * M_PI));
		coords b_end = density_to_bloch(rho, fd.f_modes_0 * pow(fd.f_energies, n_period));
		cout << "Final Bloch Vector:" << endl;
		cout << b_end << endl;
		cout << pow(fd.propagators[fd.n_prop - 1], n_period) * opt.yi << endl;
		free(fd.propagators);
		fclose(f);
		//destroyOutputBuffers(buffers, opt);
	}
	#endif
	return;
}
