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
#include "utils.h"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include <vector>
#include <complex>
#include <iostream>
#include <cufft.h>
#include <cutensor.h>
#include <unordered_map>

#define _USE_MATH_DEFINES

using Eigen::Matrix3cd;
using Eigen::Matrix2cd;
using Eigen::Matrix2d;
using Eigen::Vector2d;

#define NK 11
#define NW 17

int nearest_index(double* arr, int length, double value);

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
	Matrix3cd lookup(double frequency);
	vector<quaternion> get_collapse_ops();
	vector<pair<quaternion, Spectrum>> extract();
	double dt;
private:
	int size = NW;
};

struct covMat {
	// Representation of 3x3 covariance matrix state
	float entries[3][3][2] = {{{0}}};
	__PREPROC__ void add_outer(const float (&u_real)[3], const float (&u_imag)[3], const float (&x)[3]);
	__PREPROC__ void add(covMat other);
};	

class SpectrumAggregator {
public:
	__PREPROC__ SpectrumAggregator() {
		reset();
		dc_term = {0, 0, 0};
	}
	__PREPROC__ void initialize(double (&freq)[NW], double dt);
	CovarianceSpectrum get_covariance_spectrum();
	__PREPROC__ void update(const coords& x);
	__PREPROC__ void reset();
	__PREPROC__ void compile_results(bool islast);
	__PREPROC__ void add(SpectrumAggregator other);
	int n_samples;
	int dc_samples = 0;
private:
	coords s1[NW];
	coords s2[NW];
	covMat cmat[NW];
	double w[NW];
	coords dc_term;
	double dt;
	__PREPROC__ void set_frequencies(double (&freq)[NW]);
};

__PREPROC__ int timeSeriesLength(_PREC ioutInt, _PREC h, bool padded);

#if defined(__HIPCC__) || defined(__NVCOMPILER) || defined(__NVCC__)

#ifdef _CUFFT_H_
// cuFFT API errors
static const char* _cudaGetErrorEnum(cufftResult error)
{
	switch (error)
	{
	case CUFFT_SUCCESS:
		return "CUFFT_SUCCESS";

 	case CUFFT_INVALID_PLAN:
		return "CUFFT_INVALID_PLAN";

	case CUFFT_ALLOC_FAILED:
		return "CUFFT_ALLOC_FAILED";

	case CUFFT_INVALID_TYPE:
		return "CUFFT_INVALID_TYPE";

	case CUFFT_INVALID_VALUE:
		return "CUFFT_INVALID_VALUE";

	case CUFFT_INTERNAL_ERROR:
		return "CUFFT_INTERNAL_ERROR";

	case CUFFT_EXEC_FAILED:
		return "CUFFT_EXEC_FAILED";

	case CUFFT_SETUP_FAILED:
		return "CUFFT_SETUP_FAILED";

	case CUFFT_INVALID_SIZE:
		return "CUFFT_INVALID_SIZE";

	case CUFFT_UNALIGNED_DATA:
		return "CUFFT_UNALIGNED_DATA";
	}

	return "<unknown>";
}

#define cufftSafeCall(err)  __cufftSafeCall(err, __FILE__, __LINE__)
inline void __cufftSafeCall(cufftResult err, const char *file, const int line) {
	if( CUFFT_SUCCESS != err) {
		fprintf(stderr, "CUFFT error in file '%s', line %d\n error %d: %s\nterminating!\n", file, line, err, _cudaGetErrorEnum(err));
		cudaDeviceReset();
		assert(0);
	}
}
#endif

class FFTHandler {
// Class for handling the calls to the FFT library
// Note to self: if the input and output pointers are the same, cuFFT will automatically
// use in-place transform (and knows that the data is padded)
public:
	void plan(int npts, int ntransforms, cufftType type);
	void plan(options opt);
	void transform(void* input, void* output);
	~FFTHandler();
private:
	cufftHandle my_plan;
	cufftType transform_type;
	bool planned = false;
};

__PREPROC__ bool isFloquet(options opt); // Returns True if integrator is of Floquet type
__PREPROC__ int FFTLength(_PREC ioutInt, _PREC h);
__global__ void heavisideScale(float* correlation, int nx, int ny);
void StoCspec(complex<float>* S, CovarianceSpectrum* cspec, int nf, int nt, int nsegment);
__global__ void addComplex(cufftComplex* real_part, cufftComplex* imaginary_part, int nx, int ny);

class TensorHandler {
// Class for handling the calls to cuTensor
public:
	void plan(options opt);
	void execute(cufftComplex *B);
	void getSpectrum(CovarianceSpectrum* cspec, options opt);
	~TensorHandler();
private:
	cufftComplex* Stensor;
	void* work = nullptr;
#if CUTENSOR_MAJOR >= 2
	cutensorHandle_t handle;
	cutensorPlan_t my_plan;
	cutensorOperationDescriptor_t desc;
#else
	cutensorHandle_t* handle;
	cutensorContractionPlan_t my_plan;
	cutensorContractionDescriptor_t desc;
#endif
	cutensorTensorDescriptor_t descA;
	cutensorTensorDescriptor_t descB;
	cutensorTensorDescriptor_t descC;
	cudaStream_t stream;
	uint64_t actualWorkspaceSize = 0;
	cuFloatComplex alpha;
	cuFloatComplex beta;
	bool planned = false;
};

// Handle cuTENSOR errors
#define HANDLE_ERROR(x) { const auto err = x; \
if ( err != CUTENSOR_STATUS_SUCCESS ) { \
	printf("Error: %s\n", cutensorGetErrorString(err)); exit(-1); \
} \
}

#endif

struct floquetDiagonalization {
	quaternion f_modes_0;
	quaternion f_energies;
	double frequencies[NW];
	int n_prop;
	double period;
	std::vector<quaternion> propagators;
};

__PREPROC__ void goertzel_stage_1(const coords& x, coords& s1, coords& s2, double w, double dt);
__PREPROC__ pair<coords, coords> goertzel_stage_2_vector(const coords& s1, const coords& s2, double w, double dt);

void floquet_Delta(floquetDiagonalization fd, double (&Delta)[2][2][NK]);

void floquet_Delta(quaternion f_modes_0, quaternion f_energies, vector<quaternion> propagators, int n_prop, double period, double (&Delta)[2][2][NK]);

void floquet_X(floquetDiagonalization fd, quaternion c_op, complex<double> (&X)[2][2][NK]);

void floquet_X(quaternion f_modes_0, quaternion f_energies, quaternion c_op, vector<quaternion> propagators, int n_prop, double period, complex<double> (&X)[2][2][NK]);

void floquet_master_equation_rates(floquetDiagonalization fd, vector<quaternion> c_ops, CovarianceSpectrum cspec, complex<double> (&Zeta)[2][2], complex<double> (&Omicron)[2][2]);

void floquet_master_equation_rates(floquetDiagonalization fd, quaternion c_op, Spectrum spec, double (&Delta)[2][2][NK], complex<double> (&X)[2][2][NK], complex<double> (&Gamma)[2][2][NK], complex<double> (&Zeta)[2][2], complex<double> (&Omicron)[2][2]);

void floquet_master_equation_rates(quaternion f_modes_0, quaternion f_energies, quaternion c_op, vector<quaternion> propagators, int n_prop, double period, Spectrum spec, double (&Delta)[2][2][NK], complex<double> (&X)[2][2][NK], complex<double> (&Gamma)[2][2][NK], complex<double> (&Zeta)[2][2], complex<double> (&Omicron)[2][2]);

Matrix2cd bloch_to_density(coords bloch);
Matrix2cd bloch_to_density(coords bloch, quaternion basis);
coords density_to_bloch(Matrix2cd rho);
coords density_to_bloch(Matrix2cd rho, quaternion basis);

double heaviside(double x);

void noise_transform();

#endif
