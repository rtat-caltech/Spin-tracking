#ifndef __DOUBLE3_H_INCLUDED__
#define __DOUBLE3_H_INCLUDED__

#define _PREC double

#if defined(__NVCOMPILER) || defined(__NVCC__)
#define __PREPROC__ __host__ __device__
#include <cuda_runtime.h>
#elif defined(__HIPCC__)
#define __PREPROC__ __host__ __device__
#include <hip/hip_runtime.h>
#else
#define __PREPROC__
#endif

struct coords {
	_PREC x = 0.0;
	_PREC y = 0.0;
	_PREC z = 0.0;
};

//note that both AMD and NVIDIA include definitions of the double3 and float3 structures so only when we aren't using those do we need to define them

struct outputDtype {
	_PREC t = 0.0;
	_PREC x;
	_PREC v;
	_PREC s;
};

__PREPROC__ coords operator+(const coords, const coords);
__PREPROC__ coords operator+(const coords, const _PREC);
__PREPROC__ coords operator+(const _PREC, const coords);
__PREPROC__ coords operator-(const coords, const coords);
__PREPROC__ coords operator*(const coords, const _PREC);
__PREPROC__ coords operator*(const _PREC, const coords);
__PREPROC__ coords operator*(const coords, const coords);
__PREPROC__ coords operator/(const coords, const _PREC);
__PREPROC__ coords operator/(const coords, const coords);
__PREPROC__ coords cross(const coords, const coords);
__PREPROC__ _PREC sum(const coords);
__PREPROC__ _PREC dot(const coords, const coords);
__PREPROC__ _PREC len(const coords);
__PREPROC__ coords norm(const coords);
__PREPROC__ coords fabs3(const coords);
__PREPROC__ coords max_d3(const coords, const coords);
__PREPROC__ _PREC max3(const coords);
__PREPROC__ coords sgn(const coords);

struct quaternion{
	_PREC w = 0.0;
	_PREC x = 0.0; 
	_PREC y = 0.0;
	_PREC z = 0.0;
};

__PREPROC__ quaternion operator*(const quaternion, const quaternion);
__PREPROC__ quaternion conjugate(const quaternion);
__PREPROC__ quaternion qMult(const quaternion, const quaternion);
__PREPROC__ coords qv_mult(const quaternion, const coords);
__PREPROC__ quaternion rodriguezQuat(const coords, const _PREC);
__PREPROC__ coords rodriguez(const coords, const coords);

#endif
