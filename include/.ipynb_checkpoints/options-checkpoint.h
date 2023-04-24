#ifndef __OPTIONS_H_DEFINED__
#define __OPTIONS_H_DEFINED__

#include "double3.h"

const double G_CONST = -9.8;
const double c2 = 299792458.0 * 299792458.0;

struct options{
	double3 B0 = {0.0, 0.0, 3e-6};
	double3 E = {0, 0, 75e5};
	double3 L = {0.07, 0.1, 0.4};
	double3 yi = {1.0, 0.0, 0.0};
	
	double m = 2.2*5e-27;
	double t0 = 0.0;
	double tf = 10.0;
	double rtol = 1e-12;
	double atol = 1e-12;
	double beta = 0.0;
	double uround = 1e-16;
	double safe = 0.9;
	double fac1 = 0.333;
	double fac2 = 6.0;
	double hmax = 1.0;
	double h = 0.001;
	double tc = 1e-4;
	double T = 4.2;
	double gamma = -2.078e8;
	double V = 5.0;
	double swapStepSize = 1.0-4; //above this use rotations, below this use standard RK techniques
	double gridSize = 1.0; //output histogram size in cm for the position, assumes same bin size for each dimension
	double vecBinSize = 0.1; //the width of each bin for the spin data, always from -1 to 1
	double ioutInt = 0.05; // how frequently to output the state data
	
	unsigned int nmax = 10000000;
	int integratorType = 0; //0 means DOP853, 1 means hybrid RK45 approach
	int numParticles = 1000;
	int numPerGPUBlock = 128;
	int iout = 2; //used for the DOP853 integration method, don't mess with this
	
	char dist = 'C';
	char output = 'A'; //controls the type of output being used, N means all particles, A means average, H means histogram
	
	bool gas_coll = true;
	bool diffuse = true;
	bool gravity = true;
	//these are x, y, z coordinates
};

#endif
