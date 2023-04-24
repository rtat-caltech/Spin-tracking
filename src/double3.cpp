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

__PREPROC__ double3 operator+(const double3 a, const double3 b){
	double3 out;
	out.x = a.x + b.x;
	out.y = a.y + b.y; 
	out.z = a.z + b.z;
	return out;
}

__PREPROC__ double3 operator+(const double3 a, const double b){
	double3 out;
	out.x = a.x + b;
	out.y = a.y + b;
	out.z = a.z + b;
	return out;
}

__PREPROC__ double3 operator+(const double a, const double3 b){
	return b+a;
}

__PREPROC__ double3 operator-(const double3 a, const double3 b){
	double3 out;
	out.x = a.x - b.x;
	out.y = a.y - b.y;
	out.z = a.z - b.z;
	return out;
}

__PREPROC__ double3 operator*(const double3 a, const double b){
	double3 out;
	out.x = a.x * b;
	out.y = a.y * b;
	out.z = a.z * b;
	return out;
}

__PREPROC__ double3 operator*(const double b, const double3 a){
	return a * b;
}

__PREPROC__ double3 operator*(const double3 a, const double3 b){
	double3 out;
	out.x = a.x * b.x;
	out.y = a.y * b.y;
	out.z = a.z * b.z;
	return out;
}

__PREPROC__ double3 operator/(const double3 a, const double b){
	double3 out;
	out.x = a.x / b;
	out.y = a.y / b;
	out.z = a.z / b;
	return out;
}

__PREPROC__ double3 operator/(const double3 a, const double3 b){
	double3 out;
	out.x = a.x/b.x;
	out.y = a.y/b.y;
	out.z = a.z/b.z;
	return out;
}

__PREPROC__ double3 cross(const double3 a, const double3 b){
	double3 out;
	out.x = a.y*b.z - a.z*b.y;
	out.y = a.z*b.x - a.x*b.z;
	out.z = a.x*b.y - a.y*b.x;
	return out;
}

__PREPROC__ double sum(const double3 a){
	return a.x+a.y+a.z;
}

__PREPROC__ double len(const double3 a){
	return sqrt(a.x*a.x + a.y*a.y + a.z*a.z);
}

__PREPROC__ double3 norm(const double3 a){	
	return a/len(a);
}

__PREPROC__ double3 fabs3(const double3 a){
	double3 out;
	out.x = fabs(a.x);
	out.y = fabs(a.y);
	out.z = fabs(a.z);
	return out;
}

__PREPROC__ double3 max_d3(const double3 a, const double3 b){
	double3 out;
	out.x = (a.x>b.x)?a.x:b.x;
	out.y = (a.y>b.y)?a.y:b.y;
	out.z = (a.z>b.z)?a.z:b.z;
	return out;
}

__PREPROC__ double max3(const double3 a){
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

__PREPROC__ double3 sgn(const double3 a){
	double3 out;
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

__PREPROC__ double3 qv_mult(const quaternion q1, const double3 v1){
	double3 out;
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

__PREPROC__ quaternion rodriguezQuat(const double3 k, const double dt){
	double angle = len(k);
	double3 norm = k/angle;
	double h = angle * dt;
	norm = norm * -sin(h/2.0);
	return {cos(h/2.0), norm.x, norm.y, norm.z};
}
