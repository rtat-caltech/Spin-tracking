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
    double spare; //used for the normal generator
    bool hasSpare = false; //used for the normal generator
};

#if defined(__HIPCC__) || defined(__NVCOMPILER) || defined(__NVCC__)
//these are the larger kernel calls
__global__ void initParticlesGPU(options opt, double3 *S, double3 *v, double3 *v_old,
                              double3 *pos, double3 *pos_old, double *t, double *t_old,
                              double *tf, double *dt, double *next_gas_coll_time, double *h,
                              rngState *state, size_t *n_bounce, size_t *n_coll, size_t *n_steps,
                              unsigned int *partID, bool *stopParticle, char *coll_type, char *wall_hit);
__global__ void runSimulationGPU(options opt, double3 *S, double3 *v, double3 *v_old,
                              double3 *pos, double3 *pos_old, double *t, double *t_old,
                              double *tf, double *dt, double *next_gas_coll_time, double *h,
                              rngState *state, size_t *n_bounce, size_t *n_coll, size_t *n_steps,
                              unsigned int *partID, bool *stopParticle, char *coll_type, char *wall_hit,
                              double nextTOut);
#else
void initParticlesCPU(options opt, double3 *S, double3 *v, double3 *v_old,
                              double3 *pos, double3 *pos_old, double *t, double *t_old,
                              double *tf, double *dt, double *next_gas_coll_time, double *h,
                              rngState *state, size_t *n_bounce, size_t *n_coll, size_t *n_steps,
                              unsigned int *partID, bool *stopParticle, char *coll_type, char *wall_hit);
void runSimulationCPU(options opt, double3 *S, double3 *v, double3 *v_old,
                              double3 *pos, double3 *pos_old, double *t, double *t_old,
                              double *tf, double *dt, double *next_gas_coll_time, double *h,
                              rngState *state, size_t *n_bounce, size_t *n_coll, size_t *n_steps,
                              unsigned int *partID, bool *stopParticle, char *coll_type, char *wall_hit,
                              double nextTOut);
#endif

// typedef void (*GRAD)(const double* pos, double* G);
class particle
{
public:
	particle(const options OPT){
        opt = OPT; //set the options
        //allocate the various storage spaces
        numBlocks = std::ceil((double)opt.numParticles/(double)opt.numPerGPUBlock);
        numPartsPerBlock = opt.numPerGPUBlock;
        #if defined(__HIPCC__)
        //amd gpu allocation
        hipMalloc(&S, sizeof(double3)*OPT.numParticles); //spin state
        hipMalloc(&v, sizeof(double3)*OPT.numParticles); //velocity
        hipMalloc(&v_old, sizeof(double3)*OPT.numParticles); //velocity
        hipMalloc(&pos, sizeof(double3)*OPT.numParticles); //position
        hipMalloc(&pos_old, sizeof(double3)*OPT.numParticles); //position
        hipMalloc(&t, sizeof(double)*OPT.numParticles); //time
        hipMalloc(&t_old, sizeof(double)*OPT.numParticles); //time old
        hipMalloc(&tf, sizeof(double)*OPT.numParticles); //time final
        hipMalloc(&dt, sizeof(double)*OPT.numParticles); //dt
        hipMalloc(&next_gas_coll_time, sizeof(double)*OPT.numParticles); //gas collision time
        hipMalloc(&h, sizeof(double)*OPT.numParticles); //step size
        hipMalloc(&state, sizeof(rngState)*OPT.numParticles); //rng state
        hipMalloc(&n_bounce, sizeof(size_t)*OPT.numParticles);
        hipMalloc(&n_coll, sizeof(size_t)*OPT.numParticles);
        hipMalloc(&n_steps, sizeof(size_t)*OPT.numParticles);
        hipMalloc(&partID, sizeof(unsigned int)*OPT.numParticles);
        hipMalloc(&stopParticle, sizeof(bool)*OPT.numParticles);
        hipMalloc(&coll_type, sizeof(char)*OPT.numParticles);
        hipMalloc(&wall_hit, sizeof(char)*OPT.numParticles);
        
        //now allocate the output buffers
        S_out = (double3*)malloc(sizeof(double3)*OPT.numParticles); //spin state
        v_out = (double3*)malloc(sizeof(double3)*OPT.numParticles); //velocity
        v_old_out = (double3*)malloc(sizeof(double3)*OPT.numParticles); //velocity
        pos_out = (double3*)malloc(sizeof(double3)*OPT.numParticles); //position
        pos_old_out = (double3*)malloc(sizeof(double3)*OPT.numParticles); //position
        t_out = (double*)malloc(sizeof(double)*OPT.numParticles); //time
        t_old_out = (double*)malloc(sizeof(double)*OPT.numParticles); //time old
        tf_out = (double*)malloc(sizeof(double)*OPT.numParticles); //time final
        dt_out = (double*)malloc(sizeof(double)*OPT.numParticles); //dt
        next_gas_coll_time_out = (double*)malloc(sizeof(double)*OPT.numParticles); //gas collision time
        h_out = (double*)malloc(sizeof(double)*OPT.numParticles); //step size
        state_out = (rngState*)malloc(sizeof(rngState)*OPT.numParticles); //rng state
        n_bounce_out = (size_t*)malloc(sizeof(size_t)*OPT.numParticles);
        n_coll_out = (size_t*)malloc(sizeof(size_t)*OPT.numParticles);
        n_steps_out = (size_t*)malloc(sizeof(size_t)*OPT.numParticles);
        partID_out = (unsigned int*)malloc(sizeof(unsigned int)*OPT.numParticles);
        stopParticle_out = (bool*)malloc(sizeof(bool)*OPT.numParticles);
        coll_type_out = (char*)malloc(sizeof(char)*OPT.numParticles);
        wall_hit_out = (char*)malloc(sizeof(char)*OPT.numParticles);
        
        #elif defined(__NVCOMPILER) || defined(__NVCC__)
        //nvidia gpu allocation
        cudaMalloc(&S, sizeof(double3)*OPT.numParticles); //spin state
        cudaMalloc(&v, sizeof(double3)*OPT.numParticles); //velocity
        cudaMalloc(&v_old, sizeof(double3)*OPT.numParticles); //velocity
        cudaMalloc(&pos, sizeof(double3)*OPT.numParticles); //position
        cudaMalloc(&pos_old, sizeof(double3)*OPT.numParticles); //position
        cudaMalloc(&t, sizeof(double)*OPT.numParticles); //time
        cudaMalloc(&t_old, sizeof(double)*OPT.numParticles); //time old
        cudaMalloc(&tf, sizeof(double)*OPT.numParticles); //time final
        cudaMalloc(&dt, sizeof(double)*OPT.numParticles); //dt
        cudaMalloc(&next_gas_coll_time, sizeof(double)*OPT.numParticles); //gas collision time
        cudaMalloc(&h, sizeof(double)*OPT.numParticles); //step size
        cudaMalloc(&state, sizeof(rngState)*OPT.numParticles); //rng state
        cudaMalloc(&n_bounce, sizeof(size_t)*OPT.numParticles);
        cudaMalloc(&n_coll, sizeof(size_t)*OPT.numParticles);
        cudaMalloc(&n_steps, sizeof(size_t)*OPT.numParticles);
        cudaMalloc(&partID, sizeof(unsigned int)*OPT.numParticles);
        cudaMalloc(&stopParticle, sizeof(bool)*OPT.numParticles);
        cudaMalloc(&coll_type, sizeof(char)*OPT.numParticles);
        cudaMalloc(&wall_hit, sizeof(char)*OPT.numParticles);
        
        //now allocate the output buffers
        S_out = (double3*)malloc(sizeof(double3)*OPT.numParticles); //spin state
        v_out = (double3*)malloc(sizeof(double3)*OPT.numParticles); //velocity
        v_old_out = (double3*)malloc(sizeof(double3)*OPT.numParticles); //velocity
        pos_out = (double3*)malloc(sizeof(double3)*OPT.numParticles); //position
        pos_old_out = (double3*)malloc(sizeof(double3)*OPT.numParticles); //position
        t_out = (double*)malloc(sizeof(double)*OPT.numParticles); //time
        t_old_out = (double*)malloc(sizeof(double)*OPT.numParticles); //time old
        tf_out = (double*)malloc(sizeof(double)*OPT.numParticles); //time final
        dt_out = (double*)malloc(sizeof(double)*OPT.numParticles); //dt
        next_gas_coll_time_out = (double*)malloc(sizeof(double)*OPT.numParticles); //gas collision time
        h_out = (double*)malloc(sizeof(double)*OPT.numParticles); //step size
        state_out = (rngState*)malloc(sizeof(rngState)*OPT.numParticles); //rng state
        n_bounce_out = (size_t*)malloc(sizeof(size_t)*OPT.numParticles);
        n_coll_out = (size_t*)malloc(sizeof(size_t)*OPT.numParticles);
        n_steps_out = (size_t*)malloc(sizeof(size_t)*OPT.numParticles);
        partID_out = (unsigned int*)malloc(sizeof(unsigned int)*OPT.numParticles);
        stopParticle_out = (bool*)malloc(sizeof(bool)*OPT.numParticles);
        coll_type_out = (char*)malloc(sizeof(char)*OPT.numParticles);
        wall_hit_out = (char*)malloc(sizeof(char)*OPT.numParticles);
        
        #else
        //cpu allocation
        S = (double3*)malloc(sizeof(double3)*OPT.numParticles); //spin state
        v = (double3*)malloc(sizeof(double3)*OPT.numParticles); //velocity
        v_old = (double3*)malloc(sizeof(double3)*OPT.numParticles); //velocity
        pos = (double3*)malloc(sizeof(double3)*OPT.numParticles); //position
        pos_old = (double3*)malloc(sizeof(double3)*OPT.numParticles); //position
        t = (double*)malloc(sizeof(double)*OPT.numParticles); //time
        t_old = (double*)malloc(sizeof(double)*OPT.numParticles); //time old
        tf = (double*)malloc(sizeof(double)*OPT.numParticles); //time final
        dt = (double*)malloc(sizeof(double)*OPT.numParticles); //dt
        next_gas_coll_time = (double*)malloc(sizeof(double)*OPT.numParticles); //gas collision time
        h = (double*)malloc(sizeof(double)*OPT.numParticles); //step size
        state = (rngState*)malloc(sizeof(rngState)*OPT.numParticles); //rng state
        n_bounce = (size_t*)malloc(sizeof(size_t)*OPT.numParticles);
        n_coll = (size_t*)malloc(sizeof(size_t)*OPT.numParticles);
        n_steps = (size_t*)malloc(sizeof(size_t)*OPT.numParticles);
        partID = (unsigned int*)malloc(sizeof(unsigned int)*OPT.numParticles);
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
        hipFree(stopParticle);
        hipFree(coll_type);
        hipFree(wall_hit);
        
        //free out the output buffers
        free(S_out); //spin state
        free(v_out); //velocity
        free(v_old_out);
        free(pos_out); //position
        free(pos_old_out); //position
        free(t_out); //time
        free(t_old_out); //time old
        free(tf_out); //time final
        free(dt_out); //dt
        free(next_gas_coll_time_out); //gas collision time
        free(h_out); //step size
        free(state_out); //rng state
        free(n_bounce_out);
        free(n_coll_out);
        free(n_steps_out);
        free(partID_out);
        free(stopParticle_out);
        free(coll_type_out);
        free(wall_hit_out);
        
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
        cudaFree(stopParticle);
        cudaFree(coll_type);
        cudaFree(wall_hit);
        
        //free the output buffers
        free(S_out); //spin state
        free(v_out); //velocity
        free(v_old_out);
        free(pos_out); //position
        free(pos_old_out); //position
        free(t_out); //time
        free(t_old_out); //time old
        free(tf_out); //time final
        free(dt_out); //dt
        free(next_gas_coll_time_out); //gas collision time
        free(h_out); //step size
        free(state_out); //rng state
        free(n_bounce_out);
        free(n_coll_out);
        free(n_steps_out);
        free(partID_out);
        free(stopParticle_out);
        free(coll_type_out);
        free(wall_hit_out);
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
        free(stopParticle);
        free(coll_type);
        free(wall_hit);
        #endif
    };
    void initParticles(){
        #if defined(__HIPCC__) || defined(__NVCOMPILER) || defined(__NVCC__)
        initParticlesGPU<<<numBlocks, numPartsPerBlock>>>(opt, S, v, v_old,
                              pos, pos_old, t, t_old, tf, dt, next_gas_coll_time, h,
                              state, n_bounce, n_coll, n_steps, partID, stopParticle, coll_type, wall_hit);
        #else
        initParticlesCPU(opt, S, v, v_old,
                              pos, pos_old, t, t_old, tf, dt, next_gas_coll_time, h,
                              state, n_bounce, n_coll, n_steps, partID, stopParticle, coll_type, wall_hit);
        #endif
    };
    void runSimulation(double nextTOut){
        #if defined(__HIPCC__) || defined(__NVCOMPILER) || defined(__NVCC__)
        runSimulationGPU<<<numBlocks, numPartsPerBlock>>>(opt, S, v, v_old, pos, pos_old, t, 
                            t_old, tf, dt, next_gas_coll_time, h, state, n_bounce, n_coll, n_steps,
                            partID, stopParticle, coll_type, wall_hit, nextTOut);
        #else
        runSimulationCPU(opt, S, v, v_old, pos, pos_old, t, 
                            t_old, tf, dt, next_gas_coll_time, h, state, n_bounce, n_coll, n_steps,
                            partID, stopParticle, coll_type, wall_hit, nextTOut);
        #endif
        
    
    }
    void getCurrentState(){
        #if defined(__HIPCC__)
        //move data over for AMD GPUs
        gpuErrchk(hipMemcpy(t_out, t, sizeof(double)*opt.numParticles, hipMemcpyDeviceToHost));
        gpuErrchk(hipMemcpy(pos_out, pos, sizeof(double3)*opt.numParticles, hipMemcpyDeviceToHost));
        gpuErrchk(hipMemcpy(v_out, v, sizeof(double3)*opt.numParticles, hipMemcpyDeviceToHost));
        gpuErrchk(hipMemcpy(S_out, S, sizeof(double3)*opt.numParticles, hipMemcpyDeviceToHost));
        #elif defined(__NVCOMPILER) || defined(__NVCC__)
        //move data over for Nvidia GPUs
        gpuErrchk(cudaMemcpy(t_out, t, sizeof(double)*opt.numParticles, cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(pos_out, pos, sizeof(double3)*opt.numParticles, cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(v_out, v, sizeof(double3)*opt.numParticles, cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(S_out, S, sizeof(double3)*opt.numParticles, cudaMemcpyDeviceToHost));
        #else
        //do nothing, the data is already on CPU in this case
        #endif
    }
    void outputData(FILE *f){
        getCurrentState();
        #if defined(__HIPCC__) || defined(__NVCOMPILER) || defined(__NVCC__)
        for(int i = 0; i < opt.numParticles; i++){
            fwrite(&t_out[i], sizeof(double), 1, f);
            fwrite(&pos_out[i], sizeof(double3), 1, f);
            fwrite(&v_out[i], sizeof(double3), 1, f);
            fwrite(&S_out[i], sizeof(double3), 1, f);
        }
        #else
        for(int i = 0; i < opt.numParticles; i++){
            fwrite(&t[i], sizeof(double), 1, f);
            fwrite(&pos[i], sizeof(double3), 1, f);
            fwrite(&v[i], sizeof(double3), 1, f);
            fwrite(&S[i], sizeof(double3), 1, f);
        }
        #endif
    }
private:
    options opt;
    int numPartsPerBlock;
    int numBlocks;
    
    double3 *S;
    double3 *v;
    double3 *v_old;
    double3 *pos;
    double3 *pos_old;
    double *t;
    double *t_old;
    double *tf;
    double *dt;
    double *next_gas_coll_time;
    double *h;
    //rng states
    rngState *state;
    size_t *n_bounce;
    size_t *n_coll;
    size_t *n_steps;
    unsigned int *partID;
    bool *stopParticle;
    char *coll_type;
    char *wall_hit;
    
    double3 *S_out;
    double3 *v_out;
    double3 *v_old_out;
    double3 *pos_out;
    double3 *pos_old_out;
    double *t_out;
    double *t_old_out;
    double *tf_out;
    double *dt_out;
    double *next_gas_coll_time_out;
    double *h_out;
    //rng states
    rngState *state_out;
    size_t *n_bounce_out;
    size_t *n_coll_out;
    size_t *n_steps_out;
    unsigned int *partID_out;
    bool *stopParticle_out;
    char *coll_type_out;
    char *wall_hit_out;
};

__PREPROCD__ void calc_next_collision_time(double t, double tf, double3 v, double3 pos, 
                                           double& next_gas_coll_time, double& dt, char& coll_type, 
                                           size_t &n_bounce, size_t &n_coll, bool& finished, 
                                           char& wall_hit, rngState& state, const options opt);
template <typename T> __PREPROCD__ double sgn(T val);
__PREPROCD__ void new_velocities(double3 &v, double3 &v_old, char& coll_type, char& wall_hit, 
            rngState& state);
__PREPROCD__ void move(double &t_old, double& t, double3 &pos_old, double3 &pos, double3& v, double3& v_old, double &dt);
//__PREPROCD__ outputDtype getState(unsigned int part);
//__PREPROCD__ void updateTF(int part, const double);
//rng related functions
__PREPROCD__ uint64_t rol64(const uint64_t, const int);
__PREPROCD__ void initRNG(rngState& state, unsigned long seed);
__PREPROCD__ uint64_t xoshiro256p(rngState &state);
__PREPROCD__ double uniform(rngState &state);
__PREPROCD__ double uniform(rngState &state, const double, const double);
__PREPROCD__ double normal(rngState &state, const double, const double);
__PREPROCD__ double maxboltz(rngState &state, const double);
__PREPROCD__ double unif02pi(rngState &state);
__PREPROCD__ double exponential(rngState &state, const double);



#endif
