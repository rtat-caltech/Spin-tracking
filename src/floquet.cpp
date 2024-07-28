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

__PREPROC__ void goertzel_stage_2_vector(coords& a, coords& b, const coords& s1, const coords& s2, double w, double dt) {
	//In-place version
	double angle = w * dt;
	a = s1 - cos(angle) * s2;
	b = sin(angle) * s2;
	return;
}

__PREPROC__ Matrix3cd goertzel_stage_2(const coords& s1, const coords& s2, double w, double dt) {
	coords a, b;
	goertzel_stage_2_vector(a, b, s1, s2, w, dt);
	Vector3cd c = {complex<double> (a.x, b.x), complex<double> (a.y, b.y), complex<double> (a.z, b.z)};
	Matrix3cd m = c * c.adjoint();
	return m;
}

complex<double> Spectrum::lookup(double frequency) {
	if (frequency < 0) {
		return conj(lookup(-frequency));
	}
	return power[nearest_index(frequencies, NW, frequency)];
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
	if (n_samples == 0) {
		throw domain_error("No samples were collected. Covariance spectrum cannot be normalized");
	}
	for (int k = 0; k < NW; k++) {
		variance[k] *= (dt/n_samples);
	}
}

int nearest_index(double* arr, int length, double value) {
	double bestdiff = numeric_limits<double>::infinity();
	for (int i = 0; i < length; i++) {
		double diff = abs(arr[i] - value);
		if (diff < bestdiff) {
			bestdiff = diff;
		} else {
			return i - 1;
		}
	}
	return length - 1;
}

Matrix3cd CovarianceSpectrum::lookup(double frequency) {
	if (frequency < 0) {
		return lookup(-frequency).conjugate();
	}
	return variance[nearest_index(frequencies, NW, frequency)];
}

vector<pair<quaternion, Spectrum>> CovarianceSpectrum::extract() {
	vector<pair<quaternion, Spectrum>> out;
	vector<quaternion> c_ops = get_collapse_ops();
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

vector<quaternion> CovarianceSpectrum::get_collapse_ops() {
	vector<quaternion> c_ops;
	c_ops.push_back({0, 1, 0, 0});
	c_ops.push_back({0, 0, 1, 0});
	c_ops.push_back({0, 0, 0, 1});
	return c_ops;
}

__PREPROC__ void covMat::add_outer(const float (&u_real)[3], const float (&u_imag)[3], const float (&x)[3]) {
	// Computes c + 2 x u^\dag
	for (int i =0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			entries[i][j][0] += x[i] * u_real[j];
			entries[i][j][1] += -x[i] * u_imag[j];
		}
	}
}

__PREPROC__ void covMat::add(covMat other) {
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			for (int k = 0; k < 2; k++) {
				entries[i][j][k] += other.entries[i][j][k];
			}
		}
	}
}

__PREPROC__ void SpectrumAggregator::update(const coords& x) {
	for (int i = 0; i < NW; i++) {
		goertzel_stage_1(x, s1[i], s2[i], w[i], dt);
		coords a, b;
	    goertzel_stage_2_vector(a, b, s1[i], s2[i], w[i], dt);
	    cmat[i].add_outer({(float) a.x, (float) a.y, (float) a.z}, {(float) b.x, (float) b.y, (float) b.z}, {(float) x.x, (float) x.y, (float) x.z});
	}
	n_samples += 1;
}

__PREPROC__ void SpectrumAggregator::reset() {
	for (int i = 0; i < NW; i++) {
		s1[i] = {0, 0, 0};
		s2[i] = {0, 0, 0};
		for (int j = 0; j < 3; j++) {
			for (int k = 0; k < 3; k++) {
				for (int l = 0; l < 2; l++) {
					cmat[i].entries[j][k][l] = 0;
				}
			}
		}
	}
	n_samples = 0;
}

__PREPROC__ void SpectrumAggregator::compile_results(bool islast) {
	coords a, b;
	for (int i = 0; i < NW; i++) {
		goertzel_stage_2_vector(a, b, s1[i], s2[i], w[i], dt);
	}
}

__PREPROC__ void SpectrumAggregator::add(SpectrumAggregator other) {
	for (int i = 0; i < NW; i++) {
		cmat[i].add(other.cmat[i]);
	}
	n_samples += other.n_samples;
}

__PREPROC__ CovarianceSpectrum SpectrumAggregator::get_covariance_spectrum() {
	CovarianceSpectrum spec = CovarianceSpectrum();
	spec.initialize(w, dt, n_samples);
	for (int i = 0; i < NW; i++) {
		for (int j = 0; j < 3; j++) {
			for (int k = 0; k < 3;k++) {
				spec.variance[i](j, k) += complex<double> (cmat[i].entries[j][k][0] * 2, cmat[i].entries[j][k][1] * 2);
			}
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

/* Floquet stuff */

__PREPROC__ double sign(double x) {
	return (x < 0.0)? -1.0 : 1.0;
}

__PREPROC__ double heaviside(double x) {
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

void floquet_Delta(floquetDiagonalization fd, double (&Delta)[2][2][NK]) {
	floquet_Delta(fd.f_modes_0, fd.f_energies, fd.propagators, fd.n_prop, fd.period, Delta);
}

void floquet_Delta(quaternion f_modes_0, quaternion f_energies, vector<quaternion> propagators, int n_prop, double period, double (&Delta)[2][2][NK]) {
	double omega = 2 * M_PI/period;
	double ea = atan2(f_energies.z, f_energies.w)/period;
	double eb = -atan2(f_energies.z, f_energies.w)/period;
	double es[2] = {ea, eb};
	for (int k = 0; k < NK; k++) {
		for (int i = 0; i < 2; i++) {
			for (int j = 0; j < 2; j++) {
				double f = (es[j] - es[i]) + (k - NK/2) * omega;
				Delta[i][j][k] = f;
			}
		}
	}
}

void floquet_X(floquetDiagonalization fd, quaternion c_op, complex<double> (&X)[2][2][NK]) {
	floquet_X(fd.f_modes_0, fd.f_energies, c_op, fd.propagators, fd.n_prop, fd.period, X);
}

void floquet_X(quaternion f_modes_0, quaternion f_energies, quaternion c_op, vector<quaternion> propagators, int n_prop, double period, complex<double> (&X)[2][2][NK]) {
	quaternion Xq[NK] = {0};
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
		X[0][0][k] = Xq[k].z - im_unit * Xq[k].w;
		X[1][1][k] = conj(X[0][0][k]);
		X[0][1][k] = Xq[k].x - im_unit * Xq[k].y;
		X[1][0][k] = Xq[2*kmax-k].x + im_unit * Xq[2*kmax-k].y;
	}
}

void floquet_master_equation_rates(floquetDiagonalization fd, vector<quaternion> c_ops, CovarianceSpectrum cspec, complex<double> (&Zeta)[2][2], complex<double> (&Omicron)[2][2]) {
	complex<double> X[3][2][2][NK] = {{{{0}}}};
	double Delta[2][2][NK] = {{{0}}};
	double omega = 2 * M_PI/fd.period;
	int kmax = NK/2;
	floquet_Delta(fd, Delta);
	for (int i = 0; i < c_ops.size(); i++) {
		floquet_X(fd, c_ops[i], X[i]);
	}
	for (int a = 0; a < 3; a++) {
		for (int b = 0; b < 3; b++) {
			// Now compute Zeta (A = Zeta + Zeta^T)
			for (int k = 0; k <= kmax * 2; k++) {
				for (int i = 0; i < 2; i++) {
					for (int j = 0; j < 2; j++) {
						Zeta[i][j] = Zeta[i][j] + X[a][i][j][k] * conj(X[b][i][j][k]) * cspec.lookup(Delta[i][j][k])(a, b)/2.0;
					}
				}
			}

			// Now compute Omicron
			for (int k = 0; k <= kmax * 2; k++) {
				for (int i = 0; i < 2; i++) {
					for (int j = 0; j < 2; j++) {
						//double f = (k - NK/2) * omega;
						Omicron[i][j] = Omicron[i][j] + cspec.lookup(Delta[j][j][k])(a, b)
							* X[a][i][i][k] * conj(X[b][j][j][k]);
					}
				}
			}
		}
	}
}
/*
void floquet_master_equation_rates(floquetDiagonalization fd, quaternion c_op, Spectrum spec, double (&Delta)[2][2][NK], complex<double> (&X)[2][2][NK], complex<double> (&Gamma)[2][2][NK], complex<double> (&Zeta)[2][2], complex<double> (&Omicron)[2][2]) {
	floquet_master_equation_rates(fd.f_modes_0, fd.f_energies, c_op, fd.propagators, fd.n_prop, fd.period, spec, Delta, X, Gamma, Zeta, Omicron);
}

void floquet_master_equation_rates(quaternion f_modes_0, quaternion f_energies, quaternion c_op, vector<quaternion> propagators, int n_prop, double period, Spectrum spec, double (&Delta)[2][2][NK], complex<double> (&X)[2][2][NK], complex<double> (&Gamma)[2][2][NK], complex<double> (&Zeta)[2][2], complex<double> (&Omicron)[2][2]) {
	// The Floquet tensors will be stored in Delta, X, Gamma, A
	// The inital contents of Delta, X, Gamma do not matter (and will be overwritten).
	// The newly computed A will be added to its inital contents.
	// The spectrum is defined as 2 g^2/4 \int_0^\infty <B_i(t) B_i(t + \tau)> d\tau
	// Note that the two-sided spectrum, g^2/4 \int_-\infty^\infty <B_i(t) B_i(t + \tau)> d\tau
	// is just the real part of the one-sided spectrum
	quaternion Xq[NK] = {0};
	double Xsq[2][2][NK] = {{{0}}};
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
		X[0][0][k] = Xq[k].z - im_unit * Xq[k].w;
		X[1][1][k] = conj(X[0][0][k]);
		X[0][1][k] = Xq[k].x - im_unit * Xq[k].y;
		X[1][0][k] = Xq[2*kmax-k].x + im_unit * Xq[2*kmax-k].y;

		for (int i = 0; i < 2; i++) {
			for (int j = 0; j < 2; j++) {
				Xsq[i][j][k] = norm(X[i][j][k]); // The norm is the abs() squared
			}
		}
	}

	// Now compute Gamma
	for (int k = 0; k < NK; k++) {
		for (int i = 0; i < 2; i++) {
			for (int j = 0; j < 2; j++) {
				double f = (es[j] - es[i]) + (k - NK/2) * omega;
				Delta[i][j][k] = f;
				Gamma[i][j][k] = Xsq[i][j][k] * spec.lookup(f) * 2.0 * M_PI * heaviside(f);
			}
		}
	}

	// Now compute Zeta (A = Zeta + Zeta^T)
	for (int k = 0; k <= kmax * 2; k++) {
		for (int i = 0; i < 2; i++) {
			for (int j = 0; j < 2; j++) {
				Zeta[i][j] = Zeta[i][j] + Xsq[i][j][k] * spec.lookup(Delta[i][j][k])/2.0;
			}
		}
	}

	// Now compute Omicron
	for (int k = 0; k <= kmax * 2; k++) {
		for (int i = 0; i < 2; i++) {
			for (int j = 0; j < 2; j++) {
				double f = (k - NK/2) * omega;
				Omicron[i][j] = Omicron[i][j] + spec.lookup(f)
					* X[i][i][k] * conj(X[j][j][k]);
			}
		}
	}
	return;
}
*/

Matrix2cd bloch_to_density(coords bloch) {
	quaternion basis = {1, 0, 0, 0};
	return bloch_to_density(bloch, basis);
}

Matrix2cd bloch_to_density(coords bloch, quaternion basis) {
	Matrix2cd basis_matrix = toSU2(conj(basis));
	double x = bloch.x;
	double y = bloch.y;
	double z = bloch.z;
	Matrix2cd rho;
	rho << (1 + z), (x - y * im_unit),
		(x + y * im_unit), (1 - z);
	return basis_matrix * rho * basis_matrix.adjoint()/2.0;
}

coords density_to_bloch(Matrix2cd rho) {
	quaternion basis = {1, 0, 0, 0};
	return density_to_bloch(rho, basis);
}

coords density_to_bloch(Matrix2cd rho, quaternion basis) {
	Matrix2cd sx, sy, sz;
	Matrix2cd basis_matrix = toSU2(conj(basis));
	sx << 0, 1,
		1, 0;
	sy << 0, (0.0 - 1.0 * im_unit),
		(0.0 + 1.0 * im_unit), 0;
	sz << 1, 0,
		0, -1;
	Matrix2cd rho_lab = basis_matrix.adjoint() * rho * basis_matrix;
	return coords {(rho_lab * sx).trace().real(), (rho_lab * sy).trace().real(), (rho_lab * sz).trace().real()};
}

__PREPROC__ int timeSeriesLength(_PREC ioutInt, _PREC h, bool padded) {
	// Returns length of time series, padded so that we can do in-place transform
	if (padded) {
		return (ioutInt/h/2 + 1) * 2;
	} else {
		return ioutInt/h;
	}
}

#if defined(__HIPCC__) || defined(__NVCOMPILER) || defined(__NVCC__)

void FFTHandler::plan(int npts, int ntransforms, cufftType type) {
	// Plan for a batch of 1D Fourier transforms
	// npts: Number of points in the fourier transform
	// ntransforms: Number of fourier transforms
	// transform_type: e.g. CUFFT_R2C
	transform_type = type;
	cufftSafeCall(cufftPlan1d(&my_plan, npts, type, ntransforms));
	planned = true;
}

void FFTHandler::plan(options opt) {
	plan(timeSeriesLength(opt.ioutInt, opt.h, false), 3 * opt.numParticles, CUFFT_R2C);
}

void FFTHandler::transform(void* input, void* output) {
	if (!planned) {
		cout << "Plan was not called before transform for FFTHandler" << endl;
	}
	switch(transform_type) {
	case CUFFT_R2C:
		cufftSafeCall(cufftExecR2C(my_plan, (cufftReal*) input, (cufftComplex*) output));
		break;
	case CUFFT_C2R:
		cufftSafeCall(cufftExecC2R(my_plan, (cufftComplex*) input, (cufftReal*) output));
		break;
	case CUFFT_C2C:
		cufftSafeCall(cufftExecC2C(my_plan, (cufftComplex*) input, (cufftComplex*) output, CUFFT_FORWARD));
		break;
	default:
		cout << "Invalid transform type (was FFTHandler::plan called?)" << endl;
	}
	synchronize();
}

void TensorHandler::plan(options opt) {
	// Create vector of modes
	std::vector<int> modeA{'f','a','n'};
	std::vector<int> modeB{'f','b','n'};
	std::vector<int> modeC{'f','a','b'};
	int nmodeA = modeA.size();
	int nmodeB = modeB.size();
	int nmodeC = modeC.size();
	// Extents
	std::unordered_map<int, int64_t> extent;
	extent['f'] = FFTLength(opt.ioutInt, opt.h);
	extent['a'] = 3;
	extent['b'] = 3;
	extent['n'] = opt.numParticles;

	alpha = make_cuComplex(1.0f/extent['n'], 0.0f); //Normalization factor
	beta = make_cuComplex(1.0f, 0.0f);

	// Create a vector of extents for each tensor
	std::vector<int64_t> extentC;
	for(auto mode : modeC)
		extentC.push_back(extent[mode]);
	std::vector<int64_t> extentA;
	for(auto mode : modeA)
		extentA.push_back(extent[mode]);
	std::vector<int64_t> extentB;
	for(auto mode : modeB)
		extentB.push_back(extent[mode]);

#if CUTENSOR_MAJOR >= 2
	// CUDA types
	cutensorDataType_t typeA = CUTENSOR_C_32F;
	cutensorDataType_t typeB = CUTENSOR_C_32F;
	cutensorDataType_t typeC = CUTENSOR_C_32F;
	cutensorComputeDescriptor_t descCompute = CUTENSOR_COMPUTE_DESC_32F;
	
	const uint32_t kAlignment = 128; // Alignment of the global-memory device pointers (bytes)
	gpuErrchk(cudaMallocManaged(&Stensor, sizeof(cufftComplex) * extent['f'] * extent['a'] * extent['b']));
	
	HANDLE_ERROR(cutensorCreate(&handle));
	HANDLE_ERROR(cutensorCreateTensorDescriptor(handle,
												&descA,
												nmodeA,
												extentA.data(),
												NULL,/*stride*/
												typeA, kAlignment));

	HANDLE_ERROR(cutensorCreateTensorDescriptor(handle,
												&descB,
												nmodeB,
												extentB.data(),
												NULL,/*stride*/
												typeB, kAlignment));

	HANDLE_ERROR(cutensorCreateTensorDescriptor(handle,
												&descC,
												nmodeC,
												extentC.data(),
												NULL,/*stride*/
												typeC, kAlignment));
	
	HANDLE_ERROR(cutensorCreateContraction(handle,
										   &desc,
										   descA, modeA.data(), CUTENSOR_OP_IDENTITY,
										   descB, modeB.data(), CUTENSOR_OP_CONJ,
										   descC, modeC.data(), CUTENSOR_OP_IDENTITY,
										   descC, modeC.data(),
										   descCompute));


	// Check scalar type
	cutensorDataType_t scalarType;
	HANDLE_ERROR(cutensorOperationDescriptorGetAttribute(handle,
														 desc,
														 CUTENSOR_OPERATION_DESCRIPTOR_SCALAR_TYPE,
														 (void*)&scalarType,
														 sizeof(scalarType)));
	
	assert(scalarType == CUTENSOR_C_32F);
	
	// Set algorithm
	const cutensorAlgo_t algo = CUTENSOR_ALGO_DEFAULT;
	
	cutensorPlanPreference_t planPref;
	HANDLE_ERROR(cutensorCreatePlanPreference(
					 handle,
					 &planPref,
					 algo,
					 CUTENSOR_JIT_MODE_NONE));

	// Estimate workspace size
	uint64_t workspaceSizeEstimate = 0;
	const cutensorWorksizePreference_t workspacePref = CUTENSOR_WORKSPACE_DEFAULT;
	HANDLE_ERROR(cutensorEstimateWorkspaceSize(handle,
											   desc,
											   planPref,
											   workspacePref,
											   &workspaceSizeEstimate));

	// Create contraction plan
	HANDLE_ERROR(cutensorCreatePlan(handle,
									&my_plan,
									desc,
									planPref,
									workspaceSizeEstimate));

	// query actually used workspace
	HANDLE_ERROR(cutensorPlanGetAttribute(handle,
										  my_plan,
										  CUTENSOR_PLAN_REQUIRED_WORKSPACE,
										  &actualWorkspaceSize,
										  sizeof(actualWorkspaceSize)));

	// At this point the user knows exactly how much memory is need by the operation and
	// only the smaller actual workspace needs to be allocated
	assert(actualWorkspaceSize <= workspaceSizeEstimate);

	if (actualWorkspaceSize > 0) {
		gpuErrchk(cudaMalloc(&work, actualWorkspaceSize));
		assert(uintptr_t(work) % 128 == 0); // workspace must be aligned to 128 byte-boundary
	}
#else
	// CUDA types
	cudaDataType_t typeA = CUDA_C_32F;
	cudaDataType_t typeB = CUDA_C_32F;
	cudaDataType_t typeC = CUDA_C_32F;
	cutensorComputeType_t descCompute = CUTENSOR_COMPUTE_32F;
	
	const uint32_t kAlignment = 128; // Alignment of the global-memory device pointers (bytes)
	gpuErrchk(cudaMallocManaged(&Stensor, sizeof(cufftComplex) * extent['f'] * extent['a'] * extent['b']));
	
	HANDLE_ERROR(cutensorCreate(&handle));
	HANDLE_ERROR(cutensorInitTensorDescriptor(handle,
	                                          &descA,
	                                          nmodeA,
	                                          extentA.data(),
	                                          NULL,/*stride*/
	                                          typeA, CUTENSOR_OP_IDENTITY));

	HANDLE_ERROR(cutensorInitTensorDescriptor(handle,
	                                          &descB,
	                                          nmodeB,
	                                          extentB.data(),
	                                          NULL,/*stride*/
	                                          typeB, CUTENSOR_OP_CONJ));

	HANDLE_ERROR(cutensorInitTensorDescriptor(handle,
	                                          &descC,
	                                          nmodeC,
	                                          extentC.data(),
	                                          NULL,/*stride*/
	                                          typeC, CUTENSOR_OP_IDENTITY));
	
	HANDLE_ERROR(cutensorInitContractionDescriptor(handle,
	                                               &desc,
	                                               &descA, modeA.data(), kAlignment,
	                                               &descB, modeB.data(), kAlignment,
	                                               &descC, modeC.data(), kAlignment,
	                                               &descC, modeC.data(), kAlignment,
	                                               descCompute));

	/*
	// Check scalar type
	cudaDataType_t scalarType;
	HANDLE_ERROR(cutensorOperationDescriptorGetAttribute(handle,
														 desc,
														 CUTENSOR_OPERATION_DESCRIPTOR_SCALAR_TYPE,
														 (void*)&scalarType,
														 sizeof(scalarType)));
	
	assert(scalarType == CUTENSOR_C_32F);
	*/
	
	// Set algorithm
	const cutensorAlgo_t algo = CUTENSOR_ALGO_DEFAULT;
	
	cutensorContractionFind_t find;
	HANDLE_ERROR(cutensorInitContractionFind(
					 handle,
					 &find,
					 algo));

	// Estimate workspace size
	uint64_t worksize = 0;
	const cutensorWorksizePreference_t workspacePref = CUTENSOR_WORKSPACE_RECOMMENDED;
	HANDLE_ERROR(cutensorContractionGetWorkspaceSize(handle,
	                                                 &desc,
	                                                 &find,
	                                                 workspacePref,
	                                                 &worksize));
	actualWorkspaceSize = worksize;
	// Create contraction plan
	HANDLE_ERROR(cutensorInitContractionPlan(handle,
	                                         &my_plan,
	                                         &desc,
	                                         &find,
	                                         worksize));

	if (actualWorkspaceSize > 0) {
		gpuErrchk(cudaMalloc(&work, actualWorkspaceSize));
		assert(uintptr_t(work) % 128 == 0); // workspace must be aligned to 128 byte-boundary
	}
#endif
	
	planned = true;
}

void TensorHandler::execute(cufftComplex *B) {
	if (!planned) {
		cout << "Plan was not called before execute for TensorHandler" << endl;
	}
	gpuErrchk(cudaStreamCreate(&stream));
#if CUTENSOR_MAJOR >= 2
	HANDLE_ERROR(cutensorContract(handle,
								  my_plan,
								  (void*) &alpha, B, B,
								  (void*) &beta, Stensor, Stensor,
								  work, actualWorkspaceSize, stream));
#else
	HANDLE_ERROR(cutensorContraction(handle,
	                                 &my_plan,
	                                 (void*) &alpha, B, B,
	                                 (void*) &beta, Stensor, Stensor,
	                                 work, actualWorkspaceSize, stream));
#endif
	gpuErrchk(cudaStreamDestroy(stream));
}

void TensorHandler::getSpectrum(CovarianceSpectrum* cspec, options opt) {
	int nf = FFTLength(opt.ioutInt, opt.h);
	int nt = timeSeriesLength(opt.ioutInt, opt.h, false);
	int nbatch = 9;

	FFTHandler* inverseHandler = new FFTHandler();
	inverseHandler->plan(nt, nbatch, CUFFT_C2R);
	float* correlation;
	genericMalloc<float>(&correlation, nt * nbatch);
	
	inverseHandler->transform((cufftComplex*) Stensor, (cufftReal*) correlation);
	
	int nblock = (int) (nt * nbatch/opt.numPerGPUBlock) + 1;
	heavisideScale<<<nblock, opt.numPerGPUBlock>>>(correlation, nt, nbatch);	
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );
	
	FFTHandler* forwardHandler = new FFTHandler();
	forwardHandler->plan(nt, nbatch, CUFFT_R2C);
	cufftComplex* imaginarySpectrum;
	genericMalloc<cufftComplex>(&imaginarySpectrum, nf * nbatch);
	forwardHandler->transform(correlation, imaginarySpectrum);

	addComplex<<<nblock, opt.numPerGPUBlock>>>(Stensor, imaginarySpectrum, nf, 9);
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );

	unsigned int nbytes = sizeof(complex<float>) * nf * 9;
	complex<float>* S = (complex<float>*) malloc(nbytes);
	gpuErrchk( cudaMemcpy(S, Stensor, nbytes, cudaMemcpyDeviceToHost) );

	StoCspec(S, cspec, nf, nt, (int) round((opt.tf - opt.t0)/opt.ioutInt));

	genericFree(Stensor);
	genericFree(correlation);
	genericFree(imaginarySpectrum);
	free(S);
	delete forwardHandler;
	delete inverseHandler;
}

void StoCspec(complex<float>* S, CovarianceSpectrum* cspec, int nf, int nt, int nsegment) {
	// nf = # of frequencies in FFT
	// nt = # of time samples per segment
	// nsegment = # of segments
	float df = 1.0f/(nt * cspec->dt);
	for (int i = 0; i < NW; i++) {
		int S_idx = min(round((cspec->frequencies[i])/(2*M_PI*df)), nf-1.0f);
		for (int j = 0; j < 3; j++) {
			for (int k = 0; k < 3; k++) {
				// cuTensors are column-major
				cspec->variance[i](j, k) = S[3 * nf * k + nf * j + S_idx];
			}
		}
	}
	cspec->n_samples = nt * nsegment;
}

__global__ void heavisideScale(float* correlation, int nx, int ny) {
	// Multiplies the input by a heaviside step function \Theta(ix - nx/2) * correlation[ix, iy]
	// Also applies a scaling factor 2/nt to account for FFT normalization
	unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < nx * ny) {
		unsigned int ix = i % nx;
		if (2 * ix == nx || ix == 0) {
			correlation[i] *= 1.0f/nx;
		} else if (2 * ix + 1 > nx) {
			correlation[i] = 0;
		} else {
			correlation[i] *= 2.0f/nx;
		}
	}
}

__global__ void addComplex(cufftComplex* real_part, cufftComplex* imaginary_part, int nx, int ny) {
	unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
	cuFloatComplex im_cu = make_cuComplex(0.0f, 1.0f);
	if (i < nx * ny) {
		//real_part[i] = cuCaddf(cuCmulf(imaginary_part[i], im_cu), real_part[i]);
		real_part[i] = imaginary_part[i];
	}
}

TensorHandler::~TensorHandler() {
	if (planned) {
		HANDLE_ERROR(cutensorDestroy(handle));
#if CUTENSOR_MAJOR >= 2
		HANDLE_ERROR(cutensorDestroyPlan(my_plan));
		HANDLE_ERROR(cutensorDestroyOperationDescriptor(desc));
		HANDLE_ERROR(cutensorDestroyTensorDescriptor(descA));
		HANDLE_ERROR(cutensorDestroyTensorDescriptor(descB));
		HANDLE_ERROR(cutensorDestroyTensorDescriptor(descC));
#endif
		cudaFree(Stensor);
		cudaFree(work);
	}
}

__PREPROC__ int FFTLength(_PREC ioutInt, _PREC h) {
	// Returns length of FFT (# of complex elements)
	return timeSeriesLength(ioutInt, h, true)/2;
}

__PREPROC__ bool isFloquet(options opt) {
	return opt.integratorType == 6 || opt.integratorType == 7;
}

FFTHandler::~FFTHandler() {
	if (planned) {
		cufftSafeCall( cufftDestroy(my_plan) );
	}
}

#endif
