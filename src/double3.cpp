#include "../include/double3.h"
//#include <cmath>
#include <math.h>

#if defined(__NVCOMPILER) || defined(__NVCC__)
#define __PREPROC__ __host__ __device__
#elif defined(__HIPCC__)
#define __PREPROC__ __host__ __device__
#include <hip/hip_runtime.h>
#else
#define __PREPROC__
#endif

__PREPROC__ coords operator+(const coords a, const coords b){
	coords out;
	out.x = a.x + b.x;
	out.y = a.y + b.y; 
	out.z = a.z + b.z;
	return out;
}

__PREPROC__ coords operator+(const coords a, const _PREC b){
	coords out;
	out.x = a.x + b;
	out.y = a.y + b;
	out.z = a.z + b;
	return out;
}

__PREPROC__ coords operator+(const _PREC a, const coords b){
	return b+a;
}

__PREPROC__ coords operator-(const coords a, const coords b){
	coords out;
	out.x = a.x - b.x;
	out.y = a.y - b.y;
	out.z = a.z - b.z;
	return out;
}

__PREPROC__ coords operator*(const coords a, const _PREC b){
	coords out;
	out.x = a.x * b;
	out.y = a.y * b;
	out.z = a.z * b;
	return out;
}

__PREPROC__ coords operator*(const _PREC b, const coords a){
	return a * b;
}

__PREPROC__ coords operator*(const coords a, const coords b){
	coords out;
	out.x = a.x * b.x;
	out.y = a.y * b.y;
	out.z = a.z * b.z;
	return out;
}

__PREPROC__ coords operator/(const coords a, const _PREC b){
	coords out;
	out.x = a.x / b;
	out.y = a.y / b;
	out.z = a.z / b;
	return out;
}

__PREPROC__ coords operator/(const coords a, const coords b){
	coords out;
	out.x = a.x/b.x;
	out.y = a.y/b.y;
	out.z = a.z/b.z;
	return out;
}

__PREPROC__ coords cross(const coords a, const coords b){
	coords out;
	out.x = a.y*b.z - a.z*b.y;
	out.y = a.z*b.x - a.x*b.z;
	out.z = a.x*b.y - a.y*b.x;
	return out;
}

__PREPROC__ _PREC dot(const coords a, const coords b){
	return a.x*b.x + a.y*b.y + a.z*b.z;
}

__PREPROC__ _PREC sum(const coords a){
	return a.x+a.y+a.z;
}

__PREPROC__ _PREC len(const coords a){
	return sqrt(a.x*a.x + a.y*a.y + a.z*a.z);
}

__PREPROC__ coords norm(const coords a){	
	return a/len(a);
}

__PREPROC__ coords fabs3(const coords a){
	coords out;
	out.x = fabs(a.x);
	out.y = fabs(a.y);
	out.z = fabs(a.z);
	return out;
}

__PREPROC__ coords max_d3(const coords a, const coords b){
	coords out;
	out.x = (a.x>b.x)?a.x:b.x;
	out.y = (a.y>b.y)?a.y:b.y;
	out.z = (a.z>b.z)?a.z:b.z;
	return out;
}

__PREPROC__ _PREC max3(const coords a){
	if(a.x >= a.y && a.x >= a.z)
		return a.x;
	else if(a.y >= a.x && a.y >= a.z)
		return a.y;
	else
		return a.z;
}

template <typename T>
__PREPROC__ T sgn (T val){
	return (T(0) < val) - (val < T(0));
}

__PREPROC__ coords sgn(const coords a){
	coords out;
	out.x = sgn(a.x);
	out.y = sgn(a.y);
	out.z = sgn(a.z);
	return out;
}

__PREPROC__ quaternion qConjugate(const quaternion a){
	quaternion out;
	out.w = a.w;
	out.x = -a.x;
	out.y = -a.y;
	out.z = -a.z;
	return out;
}

__PREPROC__ quaternion qMult(const quaternion q1, const quaternion q2){
	quaternion out;
	out.w = q1.w * q2.w - q1.x * q2.x - q1.y * q2.y - q1.z * q2.z;
	out.x = q1.w * q2.x + q1.x * q2.w + q1.y * q2.z - q1.z * q2.y;
	out.y = q1.w * q2.y + q1.y * q2.w + q1.z * q2.x - q1.x * q2.z;
	out.z = q1.w * q2.z + q1.z * q2.w + q1.x * q2.y - q1.y * q2.x;
	return out;
}

__PREPROC__ coords qv_mult(const quaternion q1, const coords v1){
	coords out;
	quaternion q2;
	q2.x = v1.x;
	q2.y = v1.y;
	q2.z = v1.z;
	q2 = qMult(qMult(q1, q2), qConjugate(q1));
	out.x = q2.x;
	out.y = q2.y;
	out.z = q2.z;
	return out;
}

__PREPROC__ quaternion rodriguezQuat(const coords k, const _PREC dt){
	const _PREC angle = len(k);
	coords norm = k/angle;
	const _PREC h = angle * dt;
	norm = norm * -sin(h/2.0);
	return {(_PREC)cos((double)h/2.0), norm.x, norm.y, norm.z};
}

__PREPROC__ coords rodriguez(const coords k, const coords v1){
	const _PREC angle = len(k);
	const _PREC s =  sin(angle);
	const _PREC c = cos(angle);
	return v1 * c + (cross(v1, k) * (s/angle)) + k * (dot(k, v1) * (1.0 - c)/(angle * angle));
}

