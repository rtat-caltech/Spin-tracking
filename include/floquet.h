#ifndef __FLOQUET_H_DEFINED__
#define __FLOQUET_H_DEFINED__

#if defined(_OPENMP)
#include <omp.h>
#endif
#if defined(__HIPCC__)
#define __PREPROC__ __host__ __device__
#elif defined(__NVCOMPILER) || defined(__NVCC__)
#define __PREPROC__ __host__ __device__
#include <cuda_runtime.h>
#else
#define __PREPROC__ 
#endif

#include <math.h>
#include "double3.h"
#include "quaternion.h"
#include "options.h"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include <vector>

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
	complex<double> power[NW] = {0};
	int size = NW;
	int n_samples = 0;
	complex<double> lookup(double frequency);
};

class CovarianceSpectrum {
public:
	__PREPROC__ CovarianceSpectrum()
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

struct covMat {
	// Compact representation of 3x3 covariance matrix state
	coords real_diag;
	coords imag_diag;
	coords real_off_diag;
	coords imag_off_diag;
	__PREPROC__ void add_outer(coords u_real, coords u_imag, coords v_real, coords v_imag);
	__PREPROC__ void add(covMat other);
};	

class SpectrumAggregator {
public:
	__PREPROC__ SpectrumAggregator() {
		reset();
	}
	__PREPROC__ void initialize(double (&freq)[NW], double dt);
	CovarianceSpectrum get_covariance_spectrum();
	__PREPROC__ void update(const coords& x);
	__PREPROC__ void reset();
	__PREPROC__ void add(SpectrumAggregator other);
	int n_samples;
private:
	coords s1[NW];
	coords s2[NW];
	covMat cmat[NW];
	double w[NW];
	double dt;
	__PREPROC__ void set_frequencies(double (&freq)[NW]);
};

struct floquetDiagonalization {
	quaternion f_modes_0;
	quaternion f_energies;
	double frequencies[NW];
	double dt;
	int n_prop;
	std::vector<quaternion> propagators;
};

__PREPROC__ void goertzel_stage_1(const coords& x, coords& s1, coords& s2, double w, double dt);
__PREPROC__ pair<coords, coords> goertzel_stage_2_vector(const coords& s1, const coords& s2, double w, double dt);

void floquet_master_equation_rates(floquetDiagonalization fd, quaternion c_op, double period, Spectrum spec, double (&Delta)[2][2][NK], complex<double> (&X)[2][2][NK], complex<double> (&Gamma)[2][2][NK], complex<double> (&Zeta)[2][2], complex<double> (&Omicron)[2][2]);

void floquet_master_equation_rates(quaternion f_modes_0, quaternion f_energies, quaternion c_op, vector<quaternion> propagators, int n_prop, double period, Spectrum spec, double (&Delta)[2][2][NK], complex<double> (&X)[2][2][NK], complex<double> (&Gamma)[2][2][NK], complex<double> (&Zeta)[2][2], complex<double> (&Omicron)[2][2]);

Matrix2cd bloch_to_density(coords bloch);
Matrix2cd bloch_to_density(coords bloch, quaternion basis);
coords density_to_bloch(Matrix2cd rho);
coords density_to_bloch(Matrix2cd rho, quaternion basis);

double heaviside(double x);

#endif
