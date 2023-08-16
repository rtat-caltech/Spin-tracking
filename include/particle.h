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

// typedef void (*GRAD)(const double* pos, double* G);
class particle
{
public:
	__PREPROCD__ particle(const options OPT, const unsigned long seed, const unsigned int ipart):
        partID(ipart), t(OPT.t0), S(OPT.yi), t_old(OPT.t0), tf(OPT.t0), dt(OPT.h),
        h(OPT.h), finished(false), stopParticle(false), coll_type('W'), hasSpare(false)
        {
        //this is just the splitmix64 algorithm decomposed to be out here
        //do this here to prevent needing the extra state information later on
        uint64_t state = seed + partID;
        uint64_t result = (state += 0x9E3779B97f4A7C15);
        result = (result ^ (result >> 30)) * 0xBF58476D1CE4E5B9;
        result = (result ^ (result >> 27)) * 0x94D049BB133111EB;
        result = result ^ (result >> 31);
        rngState[0] = (uint32_t)result;
        rngState[1] = (uint32_t)(result >> 32);
        result = (state += 0x9E3779B97f4A7C15);
        result = (result ^ (result >> 30)) * 0xBF58476D1CE4E5B9;
        result = (result ^ (result >> 27)) * 0x94D049BB133111EB;
        result = result ^ (result >> 31);
        rngState[2] = (uint32_t)result;
        rngState[3] = (uint32_t)(result >> 32);
        uniform(); //initial RNG to get things going, otherwise they all share the same first value which is very strange to me still
        //calculate the collision time
        pos.x = uniform()*OPT.L.x-OPT.L.x/2.0;
        pos.y = uniform()*OPT.L.y-OPT.L.y/2.0;
        pos.z = uniform()*OPT.L.z-OPT.L.z/2.0;
        pos_old = pos;
        if (OPT.gas_coll == true){
            next_gas_coll_time = exponential(OPT.tc);
        }
        else
            next_gas_coll_time = OPT.tf + 1.0;
        if (OPT.dist == 'C') {
            double3 vec;
            vec.x = normal(0.0, 1.0);
            vec.y = normal(0.0, 1.0);
            vec.z = normal(0.0, 1.0);

            double vec_norm = len(vec);
            v = OPT.V * vec/vec_norm;
        }
        else if (OPT.dist == 'M') {
            v.x = maxboltz(OPT.sqrtKT_m);
            v.y = maxboltz(OPT.sqrtKT_m);
            v.z = maxboltz(OPT.sqrtKT_m);
        }
        v_old = v;
    }
    __PREPROCD__ ~particle() {};
    __PREPROCD__ void calc_next_collision_time(const options opt);
    template <typename T> __PREPROCD__ double sgn(T val);
    __PREPROCD__ void new_velocities(const options opt);
    __PREPROCD__ void move(const options opt);
    __PREPROCD__ void step(const options opt);
    __PREPROCD__ void run(const options opt);
    __PREPROCD__ outputDtype getState();
    __PREPROCD__ void updateTF(const double);
    //rng related functions
    __PREPROCD__ uint64_t rol64(const uint64_t, const int);
    //__PREPROCD__ uint64_t splitmix64();
    //__PREPROCD__ void xorshift128_init(uint64_t);
    __PREPROCD__ uint64_t xoshiro256p();
    __PREPROCD__ double uniform();
    __PREPROCD__ double uniform(const double, const double);
    __PREPROCD__ double normal(const double, const double);
    __PREPROCD__ double maxboltz(const double);
    __PREPROCD__ double unif02pi();
    __PREPROCD__ double exponential(const double);


private:
    double3 S;
    double3 v;
    double3 v_old;
    double3 pos;
    double3 pos_old;
    double t;
    double t_old;
    double tf;
    double dt;
    double next_gas_coll_time;
    double h;
    //rng states
    uint64_t rngState[4];
    double spareRng;
    size_t n_bounce = 0;
    size_t n_coll = 0;
    size_t n_steps = 0;
    unsigned int partID;
    bool finished;
    bool stopParticle;
    char coll_type;
    char wall_hit = 'x';
    bool hasSpare;  //used for the rng
};

#endif
