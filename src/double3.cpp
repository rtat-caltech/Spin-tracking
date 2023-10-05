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

using namespace std;

ostream& operator<<(ostream& os, const double3& x) {
	os << '{' << x.x << ", " << x.y << ", " << x.z << '}';
	return os;
}

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

__PREPROC__ double dot(const double3 a, const double3 b){
	return a.x*b.x + a.y*b.y + a.z*b.z;
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
