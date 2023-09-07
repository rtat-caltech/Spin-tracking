#ifndef __PARTICLE_H_DEFINED__
#define __PARTICLE_H_DEFINED__

#include <math.h>
#include <algorithm>
#include <random>
#include <iostream>
#include <stdint.h>

#if defined(_OPENMP)
#include <omp.h>
#endif
#if defined(__HIPCC__)
#define __PREPROCD__ __device__
#elif defined(__NVCOMPILER) || defined(__NVCC__)
#define __PREPROCD__ __device__
#include <cuda_runtime.h>
#else
#define __PREPROCD__ 
#endif

#include "integrator.h"
#include "options.h"
#include "double3.h"
#include "quaternion.h"

struct spectrum{
	double frequencies[NK];
	double3 power[NK];
	int size;
	int n_samples;
};

// typedef void (*GRAD)(const double* pos, double* G);
class particle
{
	double k = 1.380649e-23;

public:
	size_t n_bounce = 0;
	size_t n_coll = 0;
	size_t n_steps = 0;
	bool finished = false;
	bool stopParticle = false;
	double lastOutput = 0.0;
	unsigned int lastIndex = 0;
	double3 s1[NK];
	double3 s2[NK];
	double w[NK];
	
	__PREPROCD__ particle(double3 y0, options OPT, unsigned long seed, unsigned int ipart) :
		L(OPT.L), m(OPT.m), dist(OPT.dist), V_init(OPT.V), t0(OPT.t0), tf(OPT.tf), 
		diffuse(OPT.diffuse), gas_coll(OPT.gas_coll), 
		gravity(OPT.gravity), pos(), sqrtKT_m(sqrt(k*OPT.T/opt.m)), max_step(OPT.hmax),
		pos_old(), v(), v_old(), B0(OPT.B0), p_interp(), v_interp(), 
		gamma(OPT.gamma), G(), opt(OPT), ipart(ipart), seed(seed),
		integrationType(OPT.integratorType), h(OPT.h)
	{
		// First do the RNG initialization
		xorshift128_init(seed + ipart);
        uniform(); //initial RNG to get things going, otherwise they all share the same first value which is very strange to me still
		S = y0;
		//calculate the collision time
		tc = 1.6e-4*m/(k*pow(OPT.T, 8));
		pos.x = uniform()*L.x-L.x/2.0;
		pos.y = uniform()*L.y-L.y/2.0;
		pos.z = uniform()*L.z-L.z/2.0;
		pos_old = pos;
		t = t0;
		if (gas_coll == true){
			next_gas_coll_time = exponential(tc);
		}
		else
			next_gas_coll_time = tf + 1.0;
		if (dist == 'C') {
			double3 vec;
			vec.x = normal(0.0, 1.0);
			vec.y = normal(0.0, 1.0);
			vec.z = normal(0.0, 1.0);

			double vec_norm = len(vec);
			v = V_init * vec/vec_norm;
		}
		else if (dist == 'M') {
			v.x = maxboltz(sqrtKT_m);
			v.y = maxboltz(sqrtKT_m);
			v.z = maxboltz(sqrtKT_m);
		}
		Vel = len(v);
		v_old = v;
	}

	__PREPROCD__ ~particle() {};
	__PREPROCD__ void calc_next_collision_time();
	template <typename T> __PREPROCD__ double sgn(T val);
	__PREPROCD__ void new_velocities();
	__PREPROCD__ void move();
	__PREPROCD__ void step();
	__PREPROCD__ void run();
	__PREPROCD__ outputDtype getState();
	__PREPROCD__ void updateTF(double);
	__PREPROCD__ double3* get_S1();
	__PREPROCD__ double3* get_S2();
    //rng related functions
    __PREPROCD__ uint64_t rol64(uint64_t, int);
    __PREPROCD__ uint64_t splitmix64();
    __PREPROCD__ void xorshift128_init(uint64_t);
    __PREPROCD__ uint64_t xoshiro256p();
    __PREPROCD__ double uniform();
    __PREPROCD__ double uniform(double, double);
    __PREPROCD__ double normal(double, double);
    __PREPROCD__ double maxboltz(const double);
    __PREPROCD__ double unif02pi();
    __PREPROCD__ double exponential(const double);
    

private:
	options opt;
	bool diffuse;
	bool gas_coll;
	bool gravity;
	double y = 0;
	double theta = 0;
	double phi = 0;
	double m;
	double tc;
	double3 S;
	double3 v;
	double3 v_old;
	double3 pos;
	double3 pos_old;
	double3 p_interp;
	double3 v_interp;
	double3 G;
	double sqrtKT_m;
	double V_init;
	double Vel = 0.0;
	double3 L;
	char coll_type = 'W';
	char dist = 'C';
	double t0;
	double tf;
	double t;
	double t_old;
	double dt = 0.0;
	double next_gas_coll_time;
	double dx = 0.0;
	double dy = 0.0;
	double dz = 0.0;
	double dtx = 0.0;
	double dty = 0.0;
	double dtz = 0.0;
	double tbounce = 0.0;
	char wall_hit = 'x';
	double Temp = 4.2;
	double3 B0 = {0.0, 0.0, 0.0};
	double gamma;
	unsigned int ipart;
	unsigned int icount = 0;
	unsigned long iprn;
	double max_step = 0.001;
	int integrationType = 0;//default to DOP853
	double h;//keep track of hte step size in the particle;
	
	//all the various RNG related things
	uint64_t seed = 0;
    uint64_t splitmix64_state;
    uint64_t rngState[4];
    double spareRng;
    bool hasSpare = false;
};

#endif
