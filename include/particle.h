#ifndef __PARTICLE_H_DEFINED__
#define __PARTICLE_H_DEFINED__

#include <math.h>
#include <algorithm>
#include <random>
#include <iostream>
#include <stdint.h>
#include <limits>

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

#include "utils.h"
#include "integrator.h"
#include "options.h"
#include "double3.h"
#include "quaternion.h"
#include "floquet.h"
#include "logger.h"

struct rngState{
    uint64_t x;
    uint64_t y;
    uint64_t z;
    uint64_t w;
    _PREC spare; //used for the normal generator
    bool hasSpare = false; //used for the normal generator
};

#if defined(__HIPCC__) || defined(__NVCOMPILER) || defined(__NVCC__)
//these are the larger kernel calls
__global__ void initParticlesGPU(options opt, coords *S, coords *v, coords *v_old,
								 coords *pos, coords *pos_old, _PREC *t, _PREC *t_old,
								 _PREC *tf, _PREC *dt, _PREC *next_gas_coll_time, _PREC *h,
								 rngState *state, size_t *n_bounce, size_t *n_coll, size_t *n_steps,
								 unsigned int *partID, int* failureState, bool *stopParticle, char *coll_type,
								 char *wall_hit, SpectrumAggregator* specagg, floquetDiagonalization fd);
__global__ void runSimulationGPU(options opt, coords *S, coords *v, coords *v_old,
								 coords *pos, coords *pos_old, _PREC *t, _PREC *t_old,
								 _PREC *tf, _PREC *dt, _PREC *next_gas_coll_time, _PREC *h,
								 rngState *state, size_t *n_bounce, size_t *n_coll, size_t *n_steps,
								 unsigned int *partID, int* failureState, bool *stopParticle, char *coll_type,
								 char *wall_hit, SpectrumAggregator *specagg, float *Bnoise,  _PREC nextTOut);

__global__ void spectrumSum(SpectrumAggregator *specagg, CovarianceSpectrum& cspec, options opt);
#else

void initParticlesCPU(options opt, coords *S, coords *v, coords *v_old,
					  coords *pos, coords *pos_old, _PREC *t, _PREC *t_old,
					  _PREC *tf, _PREC *dt, _PREC *next_gas_coll_time, _PREC *h,
					  rngState *state, size_t *n_bounce, size_t *n_coll, size_t *n_steps,
					  unsigned int *partID, int* failureState, bool *stopParticle, char *coll_type,
					  char *wall_hit, SpectrumAggregator *specagg, floquetDiagonalization fd);
void runSimulationCPU(options opt, coords *S, coords *v, coords *v_old,
					  coords *pos, coords *pos_old, _PREC *t, _PREC *t_old,
					  _PREC *tf, _PREC *dt, _PREC *next_gas_coll_time, _PREC *h,
					  rngState *state, size_t *n_bounce, size_t *n_coll, size_t *n_steps,
					  unsigned int *partID, int* failureState, bool *stopParticle, char *coll_type,
					  char *wall_hit, SpectrumAggregator *pspecagg, CovarianceSpectrum& cspec, float *Bnoise, _PREC nextTOut);
#endif

class particle {
public:
	particle(const options OPT);
    ~particle(){
        genericFree(S); //spin state
        genericFree(v); //velocity
        genericFree(v_old);
        genericFree(pos); //position
        genericFree(pos_old); //position
        genericFree(t); //time
        genericFree(t_old); //time old
        genericFree(tf); //time final
        genericFree(dt); //dt
        genericFree(next_gas_coll_time); //gas collision time
        genericFree(h); //step size
        genericFree(state); //rng state
        genericFree(n_bounce);
        genericFree(n_coll);
        genericFree(n_steps);
        genericFree(partID);
        genericFree(failureState);
        genericFree(stopParticle);
        genericFree(coll_type);
        genericFree(wall_hit);
		if (opt.integratorType == 6) {
			genericFree(specagg);
		}
		if (opt.integratorType == 7) {
			genericFree(Bnoise);
		}
    };
    void initParticles() {
		if (opt.integratorType == 6 || opt.integratorType == 7) {
			fd = initializeSpectra(cspec, opt);
		}
		if (opt.integratorType == 7) {
			fftHandler.plan(opt);
			tensorHandler.plan(opt);
		}
        #if defined(__HIPCC__) || defined(__NVCOMPILER) || defined(__NVCC__)
		synchronize();
	    initParticlesGPU<<<numBlocks, numPartsPerBlock>>>(opt, S, v, v_old, 
	                                                      pos, pos_old, t, t_old, tf, dt, next_gas_coll_time, h,
	                                                      state, n_bounce, n_coll, n_steps, partID, failureState, stopParticle, coll_type, wall_hit, specagg, fd);
	    synchronize();
        #else
	    initParticlesCPU(opt, S, v, v_old,
	                     pos, pos_old, t, t_old, tf, dt, next_gas_coll_time, h,
	                     state, n_bounce, n_coll, n_steps, partID, failureState, stopParticle, coll_type, wall_hit, specagg, fd);
        #endif
    };
    void outputData(Logger* log);
	void runSimulation(_PREC nextTOut);
	floquetDiagonalization initializeSpectra(CovarianceSpectrum& cspec, options OPT);
	void aggregateSpectrum(CovarianceSpectrum& cspec, int numParticles);
	SpectrumAggregator* getSpectrumAggregators();
	coords* getVelocities();
	coords floquetResults();
	void postProcess(Logger* log);
	coords spinMean();
private:
    options opt;
    int numPartsPerBlock;
    int numBlocks;

	coords *S;
    coords *v;
    coords *v_old;
    coords *pos;
    coords *pos_old;
    _PREC *t;
    _PREC *t_old;
    _PREC *tf;
    _PREC *dt;
    _PREC *next_gas_coll_time;
    _PREC *h;
    //rng states
    rngState *state;
    size_t *n_bounce;
    size_t *n_coll;
    size_t *n_steps;
    unsigned int *partID;
    int *failureState;
    bool *stopParticle;
    char *coll_type;
    char *wall_hit;
    
    coords *S_out;
    coords *v_out;
    coords *v_old_out;
    coords *pos_out;
    coords *pos_old_out;
    _PREC *t_out;
    _PREC *t_old_out;
    _PREC *tf_out;
    _PREC *dt_out;
    _PREC *next_gas_coll_time_out;
    _PREC *h_out;
    //rng states
    rngState *state_out;
    size_t *n_bounce_out;
    size_t *n_coll_out;
    size_t *n_steps_out;
    unsigned int *partID_out;
    int *failureState_out;
    bool *stopParticle_out;
    char *coll_type_out;
    char *wall_hit_out;
    //Floquet stuff
    SpectrumAggregator *specagg;
    CovarianceSpectrum cspec;
    floquetDiagonalization fd;
	FFTHandler fftHandler;
	TensorHandler tensorHandler;
	float *Bnoise;
};

__PREPROCD__ void calc_next_collision_time(_PREC t, _PREC tf, coords v, coords pos, 
                                           _PREC& next_gas_coll_time, _PREC& dt, char& coll_type, 
                                           size_t &n_bounce, size_t &n_coll, bool& finished, 
                                           char& wall_hit, rngState& state, const options opt);
template <typename T> __PREPROCD__ _PREC sgn(T val);
__PREPROCD__ void update_position_and_velocity(_PREC &t_old, _PREC &t, _PREC &dt, coords &pos_old, coords &pos, coords &v_old, coords &v, char& coll_type, char& wall_hit, rngState& state, bool &stopParticle, const options opt);
__PREPROCD__ void sanity_check(_PREC &t_old, _PREC& t, coords &pos_old, coords &pos, coords& v, coords& v_old, char& coll_type, char& wall_hit, bool &stopParticle, int &failureState, const options opt);

//rng related functions
__PREPROCD__ uint64_t rol64(const uint64_t, const int);
__PREPROCD__ void initRNG(rngState& state, unsigned long seed);
__PREPROCD__ uint64_t xoshiro256p(rngState &state);
__PREPROCD__ _PREC uniform(rngState &state);
__PREPROCD__ _PREC uniform(rngState &state, const _PREC, const _PREC);
__PREPROCD__ _PREC normal(rngState &state, const _PREC, const _PREC);
__PREPROCD__ _PREC maxboltz(rngState &state, const _PREC);
__PREPROCD__ _PREC unif02pi(rngState &state);
__PREPROCD__ _PREC exponential(rngState &state, const _PREC);



#endif
