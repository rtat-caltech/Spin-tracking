#include "../include/floquet.h"
#include <unistd.h>
#include <float.h>
#include <stdint.h>
#include <limits>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

using Eigen::Matrix3cd;
using Eigen::Vector3cd;

#if defined(__HIPCC__)
#define __PREPROC__ __host__ __device__
#elif defined(__NVCOMPILER) || defined(__NVCC__)
#define __PREPROC__ __host__ __device__
#else
#include <random>
#define __PREPROC__ 
#endif

#define LAPLACE true

using namespace std;

__PREPROC__ void goertzel_stage_1(const coords& x, coords& s1, coords& s2, double w, double dt) {
	double angle = w * dt;	
	coords new_s = x + 2 * cos(angle) * s1 - s2;
	s2 = s1;
	s1 = new_s;
	return;
}

__PREPROC__ pair<coords, coords> goertzel_stage_2_vector(const coords& s1, const coords& s2, double w, double dt) {
	double angle = w * dt;
	coords a = s1 - cos(angle) * s2;
	coords b = sin(angle) * s2;
	return pair<coords, coords>(a, b);
}

__PREPROC__ Matrix3cd goertzel_stage_2(const coords& s1, const coords& s2, double w, double dt) {
	double angle = w * dt;
	coords a = s1 - cos(angle) * s2;
	coords b = sin(angle) * s2;
	Vector3cd c = {complex<double> (a.x, b.x), complex<double> (a.y, b.y), complex<double> (a.z, b.z)};
	Matrix3cd m = c * c.adjoint();
	return m;
}

void diagonalize(Matrix3cd H) {
	/*
	SelfAdjointEigenSolver<Matrix3cd> es;
	es.computeDirect(H);
	es.eigenvalues();
	es.eigenvectors();
	*/
	return;
}

complex<double> Spectrum::lookup(double frequency) {
	if (frequency < 0) {
		return conj(lookup(-frequency));
	}
	double bestdiff = numeric_limits<double>::infinity();
	for (int i = 0; i < NW; i++) {
		double diff = abs(frequencies[i] - frequency);
		if (diff < bestdiff) {
			bestdiff = diff;
		} else {
			return power[i-1];
		}
	}
	return power[NW];
}

void CovarianceSpectrum::initialize(double (&freq)[NW], double d, int n_samp) {
	for (int i=0; i < NW; i++) {
		frequencies[i] = freq[i];
	}
	dt = d;
	n_samples = n_samp;
}

void CovarianceSpectrum::add(CovarianceSpectrum other) {
	for (int k = 0; k < NW; k++) {
		variance[k] = variance[k] + other.variance[k];
	}
	n_samples = n_samples + other.n_samples;
}

void CovarianceSpectrum::normalize() {
	for (int k = 0; k < NW; k++) {
		variance[k] *= (dt/n_samples);
	}
}

vector<pair<quaternion, Spectrum>> CovarianceSpectrum::extract() {
	vector<pair<quaternion, Spectrum>> out;
	quaternion c_ops[3];
	c_ops[0] = {0, 1, 0, 0};
	c_ops[1] = {0, 0, 1, 0};
	c_ops[2] = {0, 0, 0, 1};	
	for (int i = 0; i < 3; i++) {
		Spectrum spec;
		for (int j = 0; j < NW; j++) {
			spec.frequencies[j] = frequencies[j];
			spec.power[j] = variance[j](i, i);
		}
		pair<quaternion, Spectrum> p (c_ops[i], spec);
		out.push_back(p);
	}
	return out;
}

__PREPROC__ void covMat::add_outer(coords u_real, coords u_imag, coords v_real, coords v_imag) {
	// Computes c + u v^dag
	imag_diag = imag_diag + (u_imag * v_real - u_real * v_imag);
}


__PREPROC__ void SpectrumAggregator::update(const coords& x) {
	for (int i=0; i < NW; i++) {
		goertzel_stage_1(x, s1[i], s2[i], w[i], dt);
		pair<coords, coords> p = goertzel_stage_2_vector(s1[i], s2[i], w[i], dt);
		//cmat[i] = cmat[i] + c * d.adjoint() - d * d.adjoint()/2;
		cmat[i].add_outer(p.first, p.second, x, (coords) {0, 0, 0});
	}
	n_samples += 1;
}

__PREPROC__ void SpectrumAggregator::reset() {
	for (int i = 0; i < NW; i++) {
		s1[i] = {0, 0, 0};
		s2[i] = {0, 0, 0};
		cmat[i].real_diag = {0, 0, 0};
		cmat[i].imag_diag = {0, 0, 0};
		cmat[i].real_off_diag = {0, 0, 0};
		cmat[i].imag_off_diag = {0, 0, 0};		
	}
	n_samples = 0;
}

__PREPROC__ CovarianceSpectrum SpectrumAggregator::get_covariance_spectrum() {
	CovarianceSpectrum spec;
	spec.initialize(w, dt, n_samples);
	for (int i=0; i < NW; i++) {
		if (LAPLACE) {
			spec.variance[i] = goertzel_stage_2(s1[i], s2[i], w[i], dt);
			spec.variance[i](0, 0) += complex<double> (0, cmat[i].imag_diag.x);
			spec.variance[i](1, 1) += complex<double> (0, cmat[i].imag_diag.y);
			spec.variance[i](2, 2) += complex<double> (0, cmat[i].imag_diag.z);
		} else {
			spec.variance[i] = goertzel_stage_2(s1[i], s2[i], w[i], dt);
		}
	}
	return spec;
}

__PREPROC__ void SpectrumAggregator::initialize(double (&freq)[NW], double h) {
	set_frequencies(freq);
	dt = h;
}

__PREPROC__ void SpectrumAggregator::set_frequencies(double (&freq)[NW]) {
	for (int i=0; i < NW; i++) {
		w[i] = freq[i];
	}
}

Spectrum diagonalizeSpectrum(CovarianceSpectrum& spec) {
	Spectrum s;
	return s;
}


/* Floquet stuff */

double sign(double x) {
	return (x < 0.0)? -1.0 : 1.0;
}

double heaviside(double x) {
	if (x == 0.0) {
		return 0.5;
	} else {
		return (sign(x)+1)/2;
	}
}

void print_as_su2(quaternion q) {
	cout << "[" << q.w << "+" << q.z << "j" << " " << q.y << "+" << q.x << "j" << endl;
	cout << -q.y << "+" << q.x << "j" << " " << q.w << "+" << -q.z << "j" << "]" << endl;	
	return;
}

double sq(double x) {
	return x * x;
}

void floquet_master_equation_rates(floquetDiagonalization fd, quaternion c_op, double period, Spectrum spec, double (&Delta)[2][2][NK], double (&X)[2][2][NK], complex<double> (&Gamma)[2][2][NK], complex<double> (&Zeta)[2][2]) {
	floquet_master_equation_rates(fd.f_modes_0, fd.f_energies, c_op, fd.propagators, fd.n_prop, period, spec, Delta, X, Gamma, Zeta);
}

void floquet_master_equation_rates(quaternion f_modes_0, quaternion f_energies, quaternion c_op, quaternion* propagators, int n_prop, double period, Spectrum spec, double (&Delta)[2][2][NK], double (&X)[2][2][NK], complex<double> (&Gamma)[2][2][NK], complex<double> (&Zeta)[2][2]) {
	// The Floquet tensors will be stored in Delta, X, Gamma, A
	// The inital contents of Delta, X, Gamma do not matter (and will be overwritten).
	// The newly computed A will be added to its inital contents.
	quaternion Xq[NK] = {0};
	quaternion Xqr[NK] = {0};
	quaternion Xqi[NK] = {0};
	double omega = 2 * M_PI/period;
	double ea = atan2(f_energies.z, f_energies.w)/period;
	double eb = -atan2(f_energies.z, f_energies.w)/period;
	double es[2] = {ea, eb};
	int kmax = NK/2;

	for (int i = 0; i < n_prop; i++) {
		double t = (i+1) * period/n_prop;
		double weight = 1.0/n_prop;
		quaternion phase = {cos(ea * t), 0, 0, -sin(ea * t)};
		quaternion f_modes_t = propagators[i] * f_modes_0 * phase;
		quaternion q = conj(f_modes_t) * c_op * f_modes_t;
		for (int k = 0; k < NK; k++) {
			double arg = (k - kmax) * omega * t;
			quaternion k_phase = {cos(arg), 0, 0, sin(arg)};
			Xq[k] = Xq[k] + q * k_phase * weight;
		}
	}


	for (int k = 0; k <= kmax*2; k++) {
		X[0][0][k] = sq(Xq[k].w) + sq(Xq[k].z);
		X[1][1][k] = X[0][0][k];
		X[0][1][k] = sq(Xq[k].x) + sq(Xq[k].y);
		X[1][0][k] = sq(Xq[2*kmax-k].x) + sq(Xq[2*kmax-k].y);
	}

	// Now compute Gamma
	for (int k = 0; k < NK; k++) {
		for (int i = 0; i < 2; i++) {
			for (int j = 0; j < 2; j++) {
				double f = (es[j] - es[i]) + (k - NK/2) * omega;
				Delta[i][j][k] = f;
				Gamma[i][j][k] = X[i][j][k] * spec.lookup(f);
			}
		}
	}

	// Now compute Zeta (A = Zeta + Zeta^T)
	for (int k = 0; k <= kmax * 2; k++) {
		for (int i = 0; i < 2; i++) {
			for (int j = 0; j < 2; j++) {
				Zeta[i][j] = Zeta[i][j] + Gamma[i][j][k];
			}
		}
	}
	
	return;
}

Matrix2cd bloch_to_density(coords bloch) {
	quaternion basis = {1, 0, 0, 0};
	return bloch_to_density(bloch, basis);
}

Matrix2cd bloch_to_density(coords bloch, quaternion basis) {
	Matrix2cd basis_matrix = toSU2(basis);
	double x = bloch.x;
	double y = bloch.y;
	double z = bloch.z;
	Matrix2cd rho;
	rho << (1 + z), (x - y * im_unit),
		(x + y * im_unit), (1 - z);
	return basis_matrix* rho * basis_matrix.adjoint()/2.0;
}

coords density_to_bloch(Matrix2cd rho) {
	quaternion basis = {1, 0, 0, 0};
	return density_to_bloch(rho, basis);
}

coords density_to_bloch(Matrix2cd rho, quaternion basis) {
	Matrix2cd sx, sy, sz;
	Matrix2cd basis_matrix = toSU2(basis);
	sx << 0, 1,
		1, 0;
	sy << 0, (0.0 - 1.0 * im_unit),
		(0.0 + 1.0 * im_unit), 0;
	sz << 1, 0,
		0, -1;
	Matrix2cd rho_lab = basis_matrix.adjoint() * rho * basis_matrix;
	return coords {(rho_lab * sx).trace().real(), (rho_lab * sy).trace().real(), (rho_lab * sz).trace().real()};
}
