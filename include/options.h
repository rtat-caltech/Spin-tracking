#ifndef __OPTIONS_H_DEFINED__
#define __OPTIONS_H_DEFINED__

#include "double3.h"

const double G_CONST = -9.81;
const double c2 = 299792458.0 * 299792458.0;

struct options{
	double3 B0 = {3e-6, 0.0, 0.0};
	double3 E = {75e5, 0.0, 0.0};
	double3 L = {0.07, 0.1, 0.4};
	double3 yi = {0.0, 0.0, 1.0};
	double3 posHistBins = {7, 10, 40};
	
	double m = 1.20239e-26; //2.2*5e-27; 
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
    double hmin = 1.0e-8;
	double h = 0.001;
	double T = 4.2;
	double gamma = -2.038e8; //based on vince's documentation
	double V = 5.0;
	double a = 0.0; //amplitude of the spin precession pulse
	double w = 0.0; //frequency of the spin precession pulse
	double swapStepSize = 1.0-4; //above this use rotations, below this use standard RK techniques
	double ioutInt = 0.05; // how frequently to output the state data
	
	unsigned int nmax = 10000000;
	unsigned int seed = 0;//random number seed
	int integratorType = 0; //0 means DOP853, 1 means hybrid RK45 approach
	int numParticles = 1000;
	int numPerGPUBlock = 128;
	int iout = 2; //used for the DOP853 integration method, don't mess with this
	int numPhiBins = 1000;
	int numThetaBins = 1000;
	
	char dist = 'C';
	char output = 'A'; //controls the type of output being used, N means all particles, A means average, H means histogram
	
	bool gas_coll = true;
	bool diffuse = true;
	bool gravity = true;
    bool fixedStepSize = false; //do we use adaptive step size or fixed
    bool keepStepSize = false; //do we pass the last step size to the next step or reset each time
	//these are x, y, z coordinates
};

#endif
