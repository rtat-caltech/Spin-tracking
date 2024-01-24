#include "../include/quaternion.h"
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

ostream& operator<<(ostream& os, const quaternion& q) {
	os << '{' << q.w << ", " << q.x << ", " << q.y << ", " << q.z << '}';
	return os;
}

__PREPROC__ quaternion qConjugate(const quaternion a){
	quaternion out;
	out.w = a.w;
	out.x = -a.x;
	out.y = -a.y;
	out.z = -a.z;
	return out;
}

__PREPROC__ quaternion conjugate(const quaternion q) {
	return qConjugate(q);
}

__PREPROC__ quaternion conj(const quaternion q) {
	return qConjugate(q);
}

__PREPROC__ double normSquared(const quaternion q) {
	return q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z;
}

__PREPROC__ double norm(const quaternion q) {
	return sqrt(normSquared(q));
}

__PREPROC__ quaternion operator+(const quaternion q1, const quaternion q2){
	quaternion out;
	out.w = q1.w + q2.w;
	out.x = q1.x + q2.x;
	out.y = q1.y + q2.y;
	out.z = q1.z + q2.z;
	return out;
}

__PREPROC__ quaternion operator-(const quaternion q1) {
	quaternion out;
	out.w = -q1.w;
	out.x = -q1.x;
	out.y = -q1.y;
	out.z = -q1.z;
	return out;
}

__PREPROC__ quaternion operator-(const quaternion q1, const quaternion q2){
	return q1 + (-q2);
}

__PREPROC__ quaternion operator*(const quaternion q1, const quaternion q2){
	return qMult(q1, q2);
}

__PREPROC__ quaternion operator*(const quaternion q, const double a){
	quaternion out;
	out.w = q.w * a;
	out.x = q.x * a;
	out.y = q.y * a;
	out.z = q.z * a;
	return out;
}

__PREPROC__ coords operator*(const quaternion q, const coords v1){
	return qv_mult(q, v1);
}


__PREPROC__ quaternion operator*(const double a, const quaternion q){
	return q * a;
}

__PREPROC__ quaternion operator/(const quaternion q, const double a){
	return q * (1.0/a);
}

__PREPROC__ quaternion operator/(const double a, const quaternion q) {
	return (a/normSquared(q)) * conj(q);
}

__PREPROC__ quaternion operator/(const quaternion q1, const quaternion q2){
	return q1 * (1.0/q2);
}

/* Comparison operators */

__PREPROC__ bool operator==(const quaternion q1, const quaternion q2){
	return q1.w == q2.w && q1.x == q2.x && q1.y == q2.y && q1.z == q2.z;
}

__PREPROC__ bool operator!=(const quaternion q1, const quaternion q2) {
	return !(q1 == q2);
}

__PREPROC__ bool operator<(const quaternion q1, const quaternion q2) {
	return normSquared(q1) < normSquared(q2);
}

__PREPROC__ bool operator>(const quaternion q1, const quaternion q2) {
	return q2 < q1;
}

__PREPROC__ bool operator<=(const quaternion q1, const quaternion q2) {
	return !(q1 > q2);
}

__PREPROC__ bool operator>=(const quaternion q1, const quaternion q2) {
	return !(q1 < q2);
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

__PREPROC__ quaternion rodriguezQuat(const coords k, const double dt){
	double angle = len(k);
	coords norm = k/angle;
	double h = angle * dt;
	norm = norm * -sin(h/2.0);
	return quaternion(cos(h/2.0), norm.x, norm.y, norm.z);
}

__PREPROC__ coords rodriguez(const coords k, const coords v1){
	const _PREC angle = len(k);
	const _PREC s =  sin(angle);
	const _PREC c = cos(angle);
	return v1 * c + (cross(v1, k) * (s/angle)) + k * (dot(k, v1) * (1.0 - c)/(angle * angle));
}

// "Diagonalize" a quaternion, viewed as a SU(2) matrix
// i.e. a + bi + cj + dk -> {{a + bi, c + di}, {-c + di, a - bi}}
// The diagonalization is: q = cpc*

__PREPROC__ quaternion qEigenval(const quaternion q) {
	// Returns the "eigenvalues" of a quaternion q
	return quaternion(q.w, 0.0, 0.0, -sqrt(1.0 - q.w * q.w));
}

__PREPROC__ quaternion qEigenvec(const quaternion q) {
	quaternion q1 = {sqrt(1.0 - q.w * q.w) - q.z, q.y, -q.x, 0};
	return q1/norm(q1);
}

Matrix2cd toSU2(const quaternion q) {
	Matrix2cd out;
	complex<double> a, b, c, d;
	a = q.w + im_unit * q.z;
	b = q.y + im_unit * q.x;
	c = -q.y + im_unit * q.x;
	d = q.w - im_unit * q.z;
	out << a, b, c, d;
	return out;
}

quaternion pow(quaternion q, int n) {
	if (n == 0) {
		return (quaternion) {1, 0, 0, 0};
	} else if (n == 1) {
		return q;
	} else {
		int half = n/2;
		quaternion q2 = pow(q, half);
		quaternion qr = pow(q, n - half - half);
		return q2 * q2 * qr;
	}
}
