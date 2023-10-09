#ifndef __FLOQUET_H_DEFINED__
#define __FLOQUET_H_DEFINED__

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

#include <math.h>
#include "double3.h"
#include "quaternion.h"
#include "options.h"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#define _USE_MATH_DEFINES

using Eigen::Matrix3cd;
using Eigen::Matrix2cd;
using Eigen::Matrix2d;
using Eigen::Vector2d;

#define NK 11
#define NW 17

class Spectrum {
public:
	double frequencies[NW] = {0};
	double power[NW] = {0};
	int size = NW;
	int n_samples = 0;
	double lookup(double frequency);
};

class CovarianceSpectrum {
public:
	__PREPROCD__ CovarianceSpectrum()
	{
		for (int i = 0; i < NW; i++) {
			variance[i] = Matrix3cd::Constant(0.0);
		}
	}

	double frequencies[NW] = {0};
	Matrix3cd variance[NW];
	int n_samples = 0;
	void initialize(double (&freq)[NW], double d, int n_samp = 0);
	void add(CovarianceSpectrum other);
	void normalize();
	vector<pair<quaternion, Spectrum>> extract();
private:
	double dt;
	int size = NW;
};

class SpectrumAggregator {
public:
	__PREPROCD__ SpectrumAggregator()
	{
		for (int i = 0; i < NW; i++) {
			s1[i] = {0, 0, 0};
			s2[i] = {0, 0, 0};
		}
		n_samples = 0;
	}
	__PREPROCD__ void initialize(double (&freq)[NW], double dt);
	__PREPROCD__ CovarianceSpectrum get_covariance_spectrum();
	__PREPROCD__ void update(const double3& x);
	__PREPROCD__ void reset();

private:
	double3 s1[NW];
	double3 s2[NW];
	double w[NW];
	double dt;
	int n_samples;
	__PREPROCD__ void set_frequencies(double (&freq)[NW]);
};

struct floquetDiagonalization {
	quaternion f_modes_0;
	quaternion f_energies;
	quaternion* propagators;
	int n_prop;
};

void floquet_master_equation_rates(floquetDiagonalization fd, quaternion c_op, double period, Spectrum spec, double (&Delta)[2][2][NK], double (&X)[2][2][NK], double (&Gamma)[2][2][NK], double (&A)[2][2]);

void floquet_master_equation_rates(quaternion f_modes_0, quaternion f_energies, quaternion c_op, quaternion* propagators, int n_prop, double period, Spectrum spec, double (&Delta)[2][2][NK], double (&X)[2][2][NK], double (&Gamma)[2][2][NK], double (&A)[2][2]);

Matrix2cd bloch_to_density(double3 bloch);
Matrix2cd bloch_to_density(double3 bloch, quaternion basis);
double3 density_to_bloch(Matrix2cd rho);
double3 density_to_bloch(Matrix2cd rho, quaternion basis);

#endif
