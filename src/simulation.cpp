#include "../include/simulation.h"
#include <unistd.h>
#include <chrono>

floquetDiagonalization initializeSpectra(particle* particles, CovarianceSpectrum& cspec, options OPT) {
	double t0 = 0.0;
	double tf = (2*M_PI)/OPT.w; //TODO
	int n_prop = 100;
	quaternion* propagators = (quaternion*) malloc(sizeof(quaternion) * n_prop);
	quaternion y = {1, 0, 0, 0};
	double h = 1e-6;
	for (int i=0; i < n_prop; i++) {
		double t1 = t0 + (tf - t0) * i/n_prop;
		double t2 = t0 + (tf - t0) * (i+1)/n_prop;
		integrateHamiltonian(t1, t2, y, OPT, h);
		propagators[i] = y;
	}
	
	quaternion eigen_values = qEigenval(propagators[n_prop-1]);
	quaternion eigen_vectors = qEigenvec(propagators[n_prop-1]);

	double ea = atan2(eigen_values.x, eigen_values.w);
	double eb = -atan2(eigen_values.x, eigen_values.w);
	double deltaE = ea - eb;

	double frequencies[NW];
	int count = 0;
	for(int k=0; k < NK/2; k++) {
		for (int i=-1; i < 2; i++) {
			double w = deltaE * i + k * OPT.w;
			if (w >= 0) {
				frequencies[k*3 + i] = w;
				count++;
			}
		}
	}
	for(unsigned int tid = 0; tid < OPT.numParticles; tid++){
		particles[tid].specagg.initialize(frequencies, (tf - t0)/n_prop);
	}
	cspec.initialize(frequencies, (tf - t0)/n_prop);

	floquetDiagonalization fd;
	fd.propagators = propagators;
	fd.f_modes_0 = eigen_vectors;
	fd.f_energies = eigen_values;
	fd.n_prop = n_prop;
	return fd;
}


void aggregateSpectrum(particle* particles, CovarianceSpectrum& cspec, int numParticles) {
	for(unsigned int tid = 0; tid < numParticles; tid++){
		cspec.add(particles[tid].specagg.get_covariance_spectrum());
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
		floquetDiagonalization fd = initializeSpectra(particles, cspec, opt);
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
				aggregateSpectrum(particles, cspec, opt.numParticles);				
} else {
p.outputDta(f);
				handleOutput(f, particles, opt, buffers);
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
		double3 b_end = density_to_bloch(rho, fd.f_modes_0 * pow(fd.f_energies, n_period));
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
