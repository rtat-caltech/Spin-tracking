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

#if defined(__NVCC__) || defined(__NVCOMPILER)
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
#elif defined(__HIPCC__)
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(hipError_t code, const char *file, int line, bool abort=true)
{
   if (code != hipSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", hipGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
#endif

#include "integrator.h"
#include "options.h"
#include "double3.h"

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
                              unsigned int *partID, int* failureState, bool *stopParticle, char *coll_type, char *wall_hit);
__global__ void runSimulationGPU(options opt, coords *S, coords *v, coords *v_old,
                              coords *pos, coords *pos_old, _PREC *t, _PREC *t_old,
                              _PREC *tf, _PREC *dt, _PREC *next_gas_coll_time, _PREC *h,
                              rngState *state, size_t *n_bounce, size_t *n_coll, size_t *n_steps,
                              unsigned int *partID, int* failureState, bool *stopParticle, char *coll_type, char *wall_hit, _PREC nextTOut);
#else

void initParticlesCPU(options opt, coords *S, coords *v, coords *v_old,
                              coords *pos, coords *pos_old, _PREC *t, _PREC *t_old,
                              _PREC *tf, _PREC *dt, _PREC *next_gas_coll_time, _PREC *h,
                              rngState *state, size_t *n_bounce, size_t *n_coll, size_t *n_steps,
                              unsigned int *partID, int* failureState, bool *stopParticle, char *coll_type, char *wall_hit);
void runSimulationCPU(options opt, coords *S, coords *v, coords *v_old,
                              coords *pos, coords *pos_old, _PREC *t, _PREC *t_old,
                              _PREC *tf, _PREC *dt, _PREC *next_gas_coll_time, _PREC *h,
                              rngState *state, size_t *n_bounce, size_t *n_coll, size_t *n_steps,
                              unsigned int *partID, int* failureState, bool *stopParticle, char *coll_type, char *wall_hit, _PREC nextTOut);
#endif

class particle
{
public:
	particle(const options OPT){
        opt = OPT; //set the options
        //allocate the various storage spaces
        numBlocks = std::ceil((_PREC)opt.numParticles/(_PREC)opt.numPerGPUBlock);
        numPartsPerBlock = opt.numPerGPUBlock;
        #if defined(__HIPCC__)
        //amd gpu allocation
        hipMallocManaged(&S, sizeof(coords)*OPT.numParticles); //spin state
        hipMallocManaged(&v, sizeof(coords)*OPT.numParticles); //velocity
        hipMallocManaged(&v_old, sizeof(coords)*OPT.numParticles); //velocity
        hipMallocManaged(&pos, sizeof(coords)*OPT.numParticles); //position
        hipMallocManaged(&pos_old, sizeof(coords)*OPT.numParticles); //position
        hipMallocManaged(&t, sizeof(_PREC)*OPT.numParticles); //time
        hipMallocManaged(&t_old, sizeof(_PREC)*OPT.numParticles); //time old
        hipMallocManaged(&tf, sizeof(_PREC)*OPT.numParticles); //time final
        hipMallocManaged(&dt, sizeof(_PREC)*OPT.numParticles); //dt
        hipMallocManaged(&next_gas_coll_time, sizeof(_PREC)*OPT.numParticles); //gas collision time
        hipMallocManaged(&h, sizeof(_PREC)*OPT.numParticles); //step size
        hipMallocManaged(&state, sizeof(rngState)*OPT.numParticles); //rng state
        hipMallocManaged(&n_bounce, sizeof(size_t)*OPT.numParticles);
        hipMallocManaged(&n_coll, sizeof(size_t)*OPT.numParticles);
        hipMallocManaged(&n_steps, sizeof(size_t)*OPT.numParticles);
        hipMallocManaged(&partID, sizeof(unsigned int)*OPT.numParticles);
        hipMallocManaged(&failureState, sizeof(int)*OPT.numParticles);
        hipMallocManaged(&stopParticle, sizeof(bool)*OPT.numParticles);
        hipMallocManaged(&coll_type, sizeof(char)*OPT.numParticles);
        hipMallocManaged(&wall_hit, sizeof(char)*OPT.numParticles);
        
        #elif defined(__NVCOMPILER) || defined(__NVCC__)
        //nvidia gpu allocation
        cudaMallocManaged(&S, sizeof(coords)*OPT.numParticles); //spin state
        cudaMallocManaged(&v, sizeof(coords)*OPT.numParticles); //velocity
        cudaMallocManaged(&v_old, sizeof(coords)*OPT.numParticles); //velocity
        cudaMallocManaged(&pos, sizeof(coords)*OPT.numParticles); //position
        cudaMallocManaged(&pos_old, sizeof(coords)*OPT.numParticles); //position
        cudaMallocManaged(&t, sizeof(_PREC)*OPT.numParticles); //time
        cudaMallocManaged(&t_old, sizeof(_PREC)*OPT.numParticles); //time old
        cudaMallocManaged(&tf, sizeof(_PREC)*OPT.numParticles); //time final
        cudaMallocManaged(&dt, sizeof(_PREC)*OPT.numParticles); //dt
        cudaMallocManaged(&next_gas_coll_time, sizeof(_PREC)*OPT.numParticles); //gas collision time
        cudaMallocManaged(&h, sizeof(_PREC)*OPT.numParticles); //step size
        cudaMallocManaged(&state, sizeof(rngState)*OPT.numParticles); //rng state
        cudaMallocManaged(&n_bounce, sizeof(size_t)*OPT.numParticles);
        cudaMallocManaged(&n_coll, sizeof(size_t)*OPT.numParticles);
        cudaMallocManaged(&n_steps, sizeof(size_t)*OPT.numParticles);
        cudaMallocManaged(&partID, sizeof(unsigned int)*OPT.numParticles);
        cudaMallocManaged(&failureState, sizeof(int)*OPT.numParticles);
        cudaMallocManaged(&stopParticle, sizeof(bool)*OPT.numParticles);
        cudaMallocManaged(&coll_type, sizeof(char)*OPT.numParticles);
        cudaMallocManaged(&wall_hit, sizeof(char)*OPT.numParticles);
        
        #else
        //cpu allocation
        S = (coords*)malloc(sizeof(coords)*OPT.numParticles); //spin state
        v = (coords*)malloc(sizeof(coords)*OPT.numParticles); //velocity
        v_old = (coords*)malloc(sizeof(coords)*OPT.numParticles); //velocity
        pos = (coords*)malloc(sizeof(coords)*OPT.numParticles); //position
        pos_old = (coords*)malloc(sizeof(coords)*OPT.numParticles); //position
        t = (_PREC*)malloc(sizeof(_PREC)*OPT.numParticles); //time
        t_old = (_PREC*)malloc(sizeof(_PREC)*OPT.numParticles); //time old
        tf = (_PREC*)malloc(sizeof(_PREC)*OPT.numParticles); //time final
        dt = (_PREC*)malloc(sizeof(_PREC)*OPT.numParticles); //dt
        next_gas_coll_time = (_PREC*)malloc(sizeof(_PREC)*OPT.numParticles); //gas collision time
        h = (_PREC*)malloc(sizeof(_PREC)*OPT.numParticles); //step size
        state = (rngState*)malloc(sizeof(rngState)*OPT.numParticles); //rng state
        n_bounce = (size_t*)malloc(sizeof(size_t)*OPT.numParticles);
        n_coll = (size_t*)malloc(sizeof(size_t)*OPT.numParticles);
        n_steps = (size_t*)malloc(sizeof(size_t)*OPT.numParticles);
        partID = (unsigned int*)malloc(sizeof(unsigned int)*OPT.numParticles);
        failureState = (int*)malloc(sizeof(int)*OPT.numParticles);
        stopParticle = (bool*)malloc(sizeof(bool)*OPT.numParticles);
        coll_type = (char*)malloc(sizeof(char)*OPT.numParticles);
        wall_hit = (char*)malloc(sizeof(char)*OPT.numParticles);
        #endif
    }
    ~particle(){
        #if defined(__HIPCC__)
        //amd gpu de-allocation
        hipFree(S); //spin state
        hipFree(v); //velocity
        hipFree(v_old);
        hipFree(pos); //position
        hipFree(pos_old); //position
        hipFree(t); //time
        hipFree(t_old); //time old
        hipFree(tf); //time final
        hipFree(dt); //dt
        hipFree(next_gas_coll_time); //gas collision time
        hipFree(h); //step size
        hipFree(state); //rng state
        hipFree(n_bounce);
        hipFree(n_coll);
        hipFree(n_steps);
        hipFree(partID);
        hipFree(failureState);
        hipFree(stopParticle);
        hipFree(coll_type);
        hipFree(wall_hit);
        
        #elif defined(__NVCOMPILER) || defined(__NVCC__)
        //nvidia gpu allocation
        cudaFree(S); //spin state
        cudaFree(v); //velocity
        cudaFree(v_old);
        cudaFree(pos); //position
        cudaFree(pos_old); //position
        cudaFree(t); //time
        cudaFree(t_old); //time old
        cudaFree(tf); //time final
        cudaFree(dt); //dt
        cudaFree(next_gas_coll_time); //gas collision time
        cudaFree(h); //step size
        cudaFree(state); //rng state
        cudaFree(n_bounce);
        cudaFree(n_coll);
        cudaFree(n_steps);
        cudaFree(partID);
        cudaFree(failureState);
        cudaFree(stopParticle);
        cudaFree(coll_type);
        cudaFree(wall_hit);
        
        #else
        //cpu allocation
        free(S); //spin state
        free(v); //velocity
        free(v_old);
        free(pos); //position
        free(pos_old); //position
        free(t); //time
        free(t_old); //time old
        free(tf); //time final
        free(dt); //dt
        free(next_gas_coll_time); //gas collision time
        free(h); //step size
        free(state); //rng state
        free(n_bounce);
        free(n_coll);
        free(n_steps);
        free(partID);
        free(failureState);
        free(stopParticle);
        free(coll_type);
        free(wall_hit);
        #endif
    };
    void initParticles(){
        #if defined(__HIPCC__) || defined(__NVCOMPILER) || defined(__NVCC__)
        initParticlesGPU<<<numBlocks, numPartsPerBlock>>>(opt, S, v, v_old,
                              pos, pos_old, t, t_old, tf, dt, next_gas_coll_time, h,
                              state, n_bounce, n_coll, n_steps, partID, failureState, stopParticle, coll_type, wall_hit);
        #else
        initParticlesCPU(opt, S, v, v_old,
                              pos, pos_old, t, t_old, tf, dt, next_gas_coll_time, h,
                              state, n_bounce, n_coll, n_steps, partID, failureState, stopParticle, coll_type, wall_hit);
        #endif
    };
    void runSimulation(_PREC nextTOut){
        #if defined(__HIPCC__) || defined(__NVCOMPILER) || defined(__NVCC__)
        runSimulationGPU<<<numBlocks, numPartsPerBlock>>>(opt, S, v, v_old, pos, pos_old, t, 
                            t_old, tf, dt, next_gas_coll_time, h, state, n_bounce, n_coll, n_steps,
                            partID, failureState, stopParticle,  coll_type, wall_hit, nextTOut);
        #else
        runSimulationCPU(opt, S, v, v_old, pos, pos_old, t, 
                            t_old, tf, dt, next_gas_coll_time, h, state, n_bounce, n_coll, n_steps,
                            partID, failureState, stopParticle, coll_type, wall_hit, nextTOut);
        #endif
        
    
    }
    void outputData(FILE *f){
        #if defined(__HIPCC__)
        hipDeviceSynchronize();
        #elif defined(__NVCOMPILER) || defined(__NVCC__)
        cudaDeviceSynchronize();
        #else
        
        #endif
        fwrite(t, sizeof(_PREC), opt.numParticles, f);
        fwrite(pos, sizeof(coords), opt.numParticles, f);
        fwrite(v, sizeof(coords), opt.numParticles, f);
        fwrite(S, sizeof(coords), opt.numParticles, f);
        fwrite(failureState, sizeof(int), opt.numParticles, f);
        fwrite(n_coll, sizeof(size_t), opt.numParticles, f);
        fwrite(n_bounce, sizeof(size_t), opt.numParticles, f);
        fwrite(n_steps, sizeof(size_t), opt.numParticles, f);
    }
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
    int* failureState_out;
    bool *stopParticle_out;
    char *coll_type_out;
    char *wall_hit_out;
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
