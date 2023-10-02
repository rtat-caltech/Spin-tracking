#ifndef __INTEGRATOR_H_DEFINED__
#define __INTEGRATOR_H_DEFINED__

#include <stdio.h>
#include <cmath>
#include <math.h>
#include "../include/double3.h"
#include "../include/options.h"
#include "../include/coeff.h"

#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
#include <hiprand/hiprand.h>
#include <hiprand/hiprand_kernel.h>
#define __PREPROC__ __host__ __device__
#elif defined(__NVCOMPILER) || defined(__NVCC__)
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#define __PREPROC__ __host__ __device__
#else
#include <random>
#define __PREPROC__
#endif

__PREPROC__ void obs(long nr, _PREC xold, _PREC x, coords y, coords pos, int* irtrn, 
	const options OPT, double* lastOutput, unsigned int* lastIndex, outputDtype* outputArray);		

__PREPROC__ coords pulse(const _PREC t);

__PREPROC__ coords findCrossTerm(const _PREC t, const coords& y, const options OPT, const _PREC t0, const _PREC tf ,const coords& p_old, const coords& p_new, const coords& v_old, const coords& v_new);

__PREPROC__ void Bloch(const _PREC t, const coords& y, coords& f, const options OPT, const _PREC t0, const _PREC tf ,const coords& p_old, const coords& p_new, const coords& v_old, const coords& v_new);

__PREPROC__ void interpolate(const _PREC t, const _PREC t0, const _PREC tf, const coords& p_old, const coords& p_new, const coords& v_old, const coords& v_new, coords& p_out, coords& v_out, const options OPT);

__PREPROC__ coords grad(coords&);

__PREPROC__ int integrateDOP(_PREC t0, _PREC tf, coords& y, const coords& p_old, const coords& p_new, 
	const coords& v_old, const coords& v_new, const options OPT, _PREC& h);

__PREPROC__ int integrateRK45(_PREC t0, _PREC tf, coords& y, const coords& p_old, const coords& p_new, 
	const coords& v_old, const coords& v_new, const options OPT, _PREC& h);

__PREPROC__ int integrateRKF45(_PREC t0, _PREC tf, coords& y, const coords& p_old, const coords& p_new, 
	const coords& v_old, const coords& v_new, const options OPT, _PREC& h);

__PREPROC__ int integrateRK45Quaternion(_PREC t0, _PREC tf, coords& y, const coords& p_old, 
                                        const coords& p_new, const coords& v_old, const coords& v_new, const options OPT, _PREC& h);

__PREPROC__ int integrateRKF45Quaternion(_PREC t0, _PREC tf, coords& y, const coords& p_old, 
                                        const coords& p_new, const coords& v_old, const coords& v_new, const options OPT, _PREC& h);

__PREPROC__ int integrateMagnusCFET(_PREC t0, _PREC tf, coords& y, const coords& p_old,
						const coords& p_new, const coords& v_old, const coords& v_new, const options OPT, _PREC& h);

__PREPROC__ _PREC sign(_PREC, _PREC);

__PREPROC__ _PREC min_d(_PREC, _PREC);

__PREPROC__ _PREC max_d(_PREC, _PREC);


#endif
