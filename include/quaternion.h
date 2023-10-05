#ifndef __QUATERNION_H_DEFINED__
#define __QUATERNION_H_DEFINED__

#include <utility> //So we can use pair from std
#include <iostream>
#include "double3.h"
#include <Eigen/Dense>
using Eigen::Matrix2cd;

#if defined(__NVCOMPILER) || defined(__NVCC__)
#define __PREPROC__ __host__ __device__
#include <cuda_runtime.h>
#elif defined(__HIPCC__)
#define __PREPROC__ __host__ __device__
#include <hip/hip_runtime.h>
#else
#define __PREPROC__
#endif

using namespace std;

const complex im_unit = complex<double>(0.0, 1.0);

struct quaternion{
	double w = 0.0;
	double x = 0.0; 
	double y = 0.0;
	double z = 0.0;
quaternion() : w(0.0), x(0.0), y(0.0), z(0.0) {};
quaternion(double real) : w(real), x(0.0), y(0.0), z(0.0) {};
quaternion(double a, double b, double c, double d) : w(a), x(b), y(c), z(d) {};
	friend ostream& operator<<(ostream& os, const quaternion& q);
};

__PREPROC__ bool operator==(const quaternion, const quaternion);
__PREPROC__ quaternion operator+(const quaternion, const quaternion);
__PREPROC__ quaternion operator-(const quaternion);
__PREPROC__ quaternion operator-(const quaternion, const quaternion);
__PREPROC__ quaternion operator*(const quaternion, const quaternion);
__PREPROC__ quaternion operator*(const quaternion, const double);
__PREPROC__ quaternion operator*(const double, const quaternion);
__PREPROC__ double3 operator*(const quaternion, const double3);
__PREPROC__ quaternion operator/(const quaternion, const double);
__PREPROC__ quaternion operator/(const double, const quaternion);
__PREPROC__ quaternion operator/(const quaternion, const quaternion);

__PREPROC__ bool operator<(const quaternion, const quaternion);
__PREPROC__ bool operator>(const quaternion, const quaternion);
__PREPROC__ bool operator<=(const quaternion, const quaternion);
__PREPROC__ bool operator>=(const quaternion, const quaternion);
__PREPROC__ bool operator!=(const quaternion, const quaternion);

__PREPROC__ double norm(const quaternion);
__PREPROC__ double normSquared(const quaternion); //normSquared = norm * norm
__PREPROC__ quaternion conj(const quaternion);
__PREPROC__ quaternion conjugate(const quaternion); // Different name conj(q)
__PREPROC__ quaternion qConjugate(const quaternion); // Another different name for conj(q)
__PREPROC__ quaternion qMult(const quaternion, const quaternion);
__PREPROC__ double3 qv_mult(const quaternion, const double3);
__PREPROC__ quaternion qEigenval(const quaternion);
__PREPROC__ quaternion qEigenvec(const quaternion);
__PREPROC__ quaternion rodriguezQuat(const double3, const double);
__PREPROC__ double3 rodriguez(const double3, const double3);
Matrix2cd toSU2(const quaternion);
quaternion pow(quaternion q, int n);
	
#endif
