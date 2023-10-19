#ifndef __OPTIONS_H_DEFINED__
#define __OPTIONS_H_DEFINED__

#include "double3.h"
#include "quaternion.h"

const _PREC G_CONST = -9.81;
const _PREC c2 = 299792458.0 * 299792458.0;

struct options{
	coords B0 = {3e-6, 0.0, 0.0};
	coords E = {75e5, 0.0, 0.0};
	coords L = {0.07, 0.1, 0.4};
	coords yi = {0.0, 0.0, 1.0};
	
	_PREC m = 1.20239e-26; //2.2*5e-27; 
	_PREC t0 = 0.0;
	_PREC tf = 10.0;
	_PREC rtol = 1e-12;
	_PREC atol = 1e-12;
	_PREC beta = 0.0;
	_PREC uround = 1e-16;
	_PREC safe = 0.9;
	_PREC fac1 = 0.333;
	_PREC fac2 = 6.0;
	_PREC hmax = 1.0;
	_PREC hmin = 1.0e-8;
	_PREC h = 0.001;
	_PREC T = 4.2;
    _PREC sqrtKT_m = 0.0;
    _PREC tc = 0.0;
	_PREC gamma = -2.038e8; //based on vince's documentation
	_PREC V = 5.0;
	_PREC a = 0.0; //amplitude of the spin precession pulse
	_PREC w = 0.0; //frequency of the spin precession pulse
	_PREC swapStepSize = 1.0-4; //above this use rotations, below this use standard RK techniques
    _PREC maxPosStep = 0.1; //largest step size for position/velocity integration
	_PREC ioutInt = 0.05; // how frequently to output the state data
    _PREC diffuse = 1.0; //the probability of diffuse collisions (1 meaning 100%, 0 meaning 0%)

    
	unsigned int nmax = 10000000;
	unsigned int seed = 0;//random number seed
	int integratorType = 0; //0 means DOP853, 1 means hybrid RK45 approach
	int numParticles = 1000;
	int numPerGPUBlock = 128;
	int iout = 2; //used for the DOP853 integration method, don't mess with this    
	
	char dist = 'C';
	
	bool gas_coll = true;
	bool gravity = true;
	bool fixedStepSize = false; //do we use adaptive step size or fixed
   	bool keepStepSize = false; //do we pass the last step size to the next step or reset each time
	//these are x, y, z coordinates
	bool noiseEnable = false; // Turns on the test noise function
	coords noiseAmplitudes = {0, 0, 0};
	coords noiseFrequencies = {0, 0, 0};
};

#endif
