#include <boost/test/unit_test.hpp>
#include <math.h>
#include "../include/particle.h"
#include "../include/quaternion.h"
#include "../include/double3.h"
#include "../include/floquet.h"
#include "../include/integrator.h"
#include "../include/simulation.h"
#include <iostream>
#include <stdlib.h>

namespace utf = boost::unit_test;
using namespace std;

namespace boost { namespace math { namespace fpc {
			template <>
			struct tolerance_based< quaternion > : boost::true_type{};
		} } }

bool coords_compare(coords a, coords b, double tol) {
	return len(a - b) < tol;
}

double sine_spectrum(double a, double w1, double w2, double t0, double tf, double dt) {	
	double real_part = 0.0;
	double im_part = 0.0;
	double t = first_sample_point(t0, dt);
	int n = 0;
	nsamp = (tf - t0);
	for (int i = 0; i < nsamp; i++) {
		for (int j = i; j < nsamp j++) {
			total += 
		}
	}
	/*
	while (t <= first_sample_point(tf, dt) - (dt/2)) {
		real_part += sin(w1 * t) * cos(w2 * t);
		im_part += sin(w1 * t) * sin(w2 * t);
		t += dt;
		n++;
	}
	return a * a * (real_part * real_part + im_part * im_part) * dt/n;
	*/
}

BOOST_AUTO_TEST_SUITE(FloquetIntegration)

BOOST_AUTO_TEST_CASE(diagonalization, * utf::tolerance(quaternion(1e-9))) {
	quaternion q = {2, 3, -5, 7};
	q = q/norm(q);
	quaternion eigen_values = qEigenval(q);
	quaternion eigen_vectors = qEigenvec(q);
	BOOST_TEST(eigen_values.x == 0);
	BOOST_TEST(eigen_values.y == 0);
	quaternion a = eigen_vectors * eigen_values * conj(eigen_vectors);
	BOOST_TEST(q == a);
}

BOOST_AUTO_TEST_CASE(propagators, * utf::tolerance(1e-8)) {
	double t0 = 0.0;
	double tf = 1e-3;
	options opt;
	opt.gravity = false;
	opt.T = 0.4;
	opt.B0 = {3e-6, 0.0, 0.0};
	opt.a = 3e-5;
	opt.w = 1/tf * 2 * M_PI;
	opt.atol = 1e-12;
	opt.rtol = 1e-12;
	coords s0 = {0, 1, 0};
	coords s = {0, 1, 0};

	int n_prop = 100;
	quaternion* propagators = (quaternion*) malloc(sizeof(quaternion) * n_prop);
	quaternion y = {1, 0, 0, 0};
	coords dummy = {0, 0, 0};
	double h = 1e-6;
	for (int i=0; i < n_prop; i++) {
		double t1 = t0 + (tf - t0) * i/n_prop;
		double t2 = t0 + (tf - t0) * (i+1)/n_prop;
		integrateHamiltonian(t1, t2, y, opt, 1e-6);
		integrateMagnusCFET(t1, t2, s, dummy, dummy, dummy, dummy, opt, h);
		coords a = y * s0;
		coords b = s;
	  	BOOST_TEST(a.x == b.x);
		BOOST_TEST(a.y == b.y);
		BOOST_TEST(a.z == b.z);
		propagators[i] = y;
	}
	// Look in scripts/qutipFloquet.py for how these tensors are calculated
	double Delta_ref[2][2][NK] = {{{-31415.9265359, -25132.74122872, -18849.55592154, -12566.37061436, -6283.18530718, 0., 6283.18530718, 12566.37061436, 18849.55592154, 25132.74122872, 31415.9265359}, {-30941.87521652, -24658.68990934, -18375.50460216, -12092.31929499, -5809.13398781, 474.05131937, 6757.23662655, 13040.42193373, 19323.60724091, 25606.79254809, 31889.97785527}}, {{-31889.97785527, -25606.79254809, -19323.60724091, -13040.42193373, -6757.23662655, -474.05131937, 5809.13398781, 12092.31929499, 18375.50460216, 24658.68990934, 30941.87521652}, {-31415.9265359, -25132.74122872, -18849.55592154, -12566.37061436, -6283.18530718, 0., 6283.18530718, 12566.37061436, 18849.55592154, 25132.74122872, 31415.9265359}}};

	double X_ref[2][2][NK] = {{{4.82902543e-08, 1.42500174e-29, 3.30960564e-04, 9.12919515e-29, 1.87688610e-01, 1.42211545e-28, 1.87688610e-01, 9.12898629e-29, 3.30960564e-04, 1.42716098e-29, 4.82902543e-08}, {3.15136984e-29, 7.15550842e-06, 1.27178379e-28, 1.57109229e-02, 1.07067023e-26, 5.98999978e-01, 4.95743979e-27, 9.23919626e-03, 2.17816906e-28, 3.50971835e-06, 2.10895130e-29}}, {{2.10986163e-29, 3.50971835e-06, 2.17820263e-28, 9.23919626e-03, 4.95687493e-27, 5.98999978e-01, 1.07096047e-26, 1.57109229e-02, 1.27314158e-28, 7.15550842e-06, 3.12693008e-29}, {4.82902544e-08, 6.83511992e-28, 3.30960564e-04, 3.54274315e-27, 1.87688610e-01, 1.09678335e-26, 1.87688610e-01, 3.54106031e-27, 3.30960564e-04, 6.83481738e-28, 4.82902544e-08}}};
	
	double X_re_ref[2][2][NK] = {{{-5.62375667e-16, 1.12019768e-15, -8.00238781e-15, 3.56463990e-15, -7.13459240e-15, 1.19252481e-14, -7.14847018e-15, 3.56463990e-15, -8.00130361e-15, 1.11965558e-15, -5.61941986e-16}, {-2.64942514e-15, 1.06418386e-14, 5.79204954e-15, 1.36181017e-14, -3.61154889e-14, -4.43572181e-15, 3.46068216e-14, -8.31895517e-15, -3.87040961e-15, -8.54152220e-15, 2.62297060e-15}}, {{2.62470533e-15, -8.54152220e-15, -3.87084329e-15, -8.33977185e-15, 3.45925101e-14, -4.43462405e-15, -3.61742527e-14, 1.36285101e-14, 5.80375892e-15, 1.06435734e-14, -2.64855778e-15}, {-7.43047117e-15, -1.25160299e-14, 2.13183421e-14, -7.20354765e-15, 6.31592218e-14, -1.04727425e-13, 6.31592218e-14, -7.20007821e-15, 2.13189926e-14, -1.25151625e-14, -7.42938697e-15}}};
	
	double X_im_ref[2][2][NK] = {{{2.19750437e-04, -3.60488205e-15, 1.81923216e-02, -8.86483468e-15, 4.33230435e-01, -1.44712250e-18, -4.33230435e-01, 8.86471687e-15, -1.81923216e-02, 3.60804396e-15, -2.19750437e-04}, {4.94916608e-15, -2.67497821e-03, 9.67628755e-15, -1.25343221e-01, -9.69658381e-14, -7.73950888e-01, 6.13172707e-14, -9.61207379e-02, 1.42420797e-14, -1.87342423e-03, 3.76955411e-15}}, {{-3.76955411e-15, 1.87342423e-03, -1.42420797e-14, 9.61207379e-02, -6.13207402e-14, 7.73950888e-01, 9.69588992e-14, 1.25343221e-01, -9.67628755e-15, 2.67497821e-03, -4.92487995e-15}, {-2.19750437e-04, 2.29534526e-14, -1.81923216e-02, 5.90834329e-14, -4.33230435e-01, -2.62499347e-18, 4.33230435e-01, -5.90696130e-14, 1.81923216e-02, -2.29532665e-14, 2.19750437e-04}}};

	double Gamma_ref[2][2][NK] = {{{0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 4.46770744e-28, 2.91336071e-02, 3.60945112e-30, 5.83624178e-06, 1.41737937e-31, 3.07114136e-10}, {0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 3.07304001e+00, 6.67559967e-28, 3.39378535e-04, 3.65538839e-30, 3.35799422e-08, 1.30170073e-31}}, {{0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.93663610e-27, 6.70505594e-04, 2.36207527e-30, 7.38187545e-08, 2.04998836e-31}, {0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 3.44564651e-26, 2.91336071e-02, 1.40007704e-28, 5.83624178e-06, 6.78797223e-30, 3.07114136e-10}}};
	
	double A_ref[2][2] = {{2.91394437e-02, 3.07337942e+00}, {6.70579412e-04, 2.91394437e-02}};
	
	coords b_ref = {0., 0.99007903, 0.};

	double Delta[2][2][NK] = {{{0}}};
	complex<double> X[2][2][NK] = {{{0}}};
	complex<double> Gamma[2][2][NK] = {{{0}}};
	complex<double> Zeta[2][2] = {{0}};
	complex<double> Omicron[2][2] = {{0}};
	
	quaternion eigen_values = qEigenval(propagators[n_prop-1]);
	quaternion eigen_vectors = qEigenvec(propagators[n_prop-1]);

	double ea = -atan2(eigen_values.z, eigen_values.w)/(tf - t0);
	double eb = atan2(eigen_values.z, eigen_values.w)/(tf - t0);
	double deltaE = (ea - eb);

	quaternion c_op = {0, 0, 1, 0};
	Spectrum spec;
	for(int k=0; k <= NK/2; k++) {
		for (int i=-1; i < 2; i++) {
			double w = deltaE * i + k * opt.w;
			if (w < 0) {
				continue;
			}			
			spec.frequencies[k*3 + i] = w;
			spec.power[k*3 + i] = 1/(pow(w * (tf - t0), 2) + 1);
		}
	}
	
	floquet_master_equation_rates(eigen_vectors, eigen_values, c_op, propagators, n_prop, tf - t0, spec, Delta, X, Gamma, Zeta, Omicron);

	for (int i=0; i < 2; i++) {
		for (int j=0; j < 2; j++) {
			// The reference A is for T=0. For T=Infinity (our case), we should use A_ref + A_ref^T
			BOOST_TEST(abs(2 * M_PI * Zeta[i][j]) == A_ref[i][j] + A_ref[j][i]);
			for (int k = 0; k < NK; k++) {
				BOOST_TEST(Delta[i][j][k] == Delta_ref[i][j][k]);
				BOOST_TEST(real(X[i][j][k]) - X_re_ref[i][j][k] == 0);
				BOOST_TEST(imag(X[i][j][k]) - X_im_ref[i][j][k] == 0);
				BOOST_TEST(abs(Gamma[i][j][k]) - Gamma_ref[i][j][k] == 0);
			}
		}
	}

	Matrix2cd rho = bloch_to_density(s0);
	rho = integrateFloquetMarkov(t0, (tf - t0) * 10 + t0, rho, Zeta, Omicron);
	coords sf = density_to_bloch(rho);
	BOOST_TEST(sf.x == b_ref.x);
	BOOST_TEST(sf.y == b_ref.y);
	BOOST_TEST(sf.z == b_ref.z);

	free(propagators);
}

BOOST_AUTO_TEST_CASE(spectrum_calculation, * utf::tolerance(1e-9)) {
	// Tests whether the noise spectrum is calculated correctly.
	options opt;
	opt.gravity = false;
	opt.T = 0.4;
	opt.a = 1.0;
	opt.w = 1e3 * 2 * M_PI;
	opt.noiseAmplitudes = {1.0, 2.0, 3.0}; // Artificially inject a sine wave B-field
	opt.noiseFrequencies = {1.0e3, 3.5e3, -3.5e3}; // See the testNoise function in integrators.cpp
	double t0 = 0.1;
	double tf = 1.2;
	coords y = (coords){0, 0, 1};
	coords p_old = (coords) {0, 0, 0};
	coords p_new = (coords) {0, 0, 0};
	coords v_old = (coords) {0, 0, 0};
	coords v_new = (coords) {0, 0, 0};
	double B0 = 3e-6;
	opt.B0 = {B0, 0.0, 0.0};
	opt.E = {0.0, 0.0, 0.0};
	opt.tf = tf;
	opt.t0 = t0;
	opt.rtol = 1e-12;
	opt.atol = 1e-12;
	opt.gravity = false;
	opt.numParticles = 1;
	double dt = 1e-5;

	SpectrumAggregator specagg;
	double w[NW] = {0};
	w[0] = opt.noiseFrequencies.x;
	w[1] = opt.noiseFrequencies.y;
	w[2] = opt.noiseFrequencies.z;
	particle p = particle(opt);
	p.initParticles();
	p.getSpectrumAggregators()[0].initialize(w, dt);
	
	int n_steps = integrateSpectrum(t0, tf, p.getSpectrumAggregators()[0], p_old, p_new, v_old, v_new, opt, dt);
	CovarianceSpectrum spec;
	spec.initialize(w, dt);
	p.aggregateSpectrum(spec, opt.numParticles);
	spec.normalize();

	BOOST_TEST(spec.variance[0](0, 0).real() == opt.gamma * opt.gamma * sine_spectrum(opt.noiseAmplitudes.x, opt.noiseFrequencies.x, opt.noiseFrequencies.x, t0, tf, dt));
	BOOST_TEST(spec.variance[1](1, 1).real() == opt.gamma * opt.gamma * sine_spectrum(opt.noiseAmplitudes.y, opt.noiseFrequencies.y, opt.noiseFrequencies.y, t0, tf, dt));
	BOOST_TEST(spec.variance[2](2, 2).real() == opt.gamma * opt.gamma * sine_spectrum(opt.noiseAmplitudes.z, opt.noiseFrequencies.z, opt.noiseFrequencies.z, t0, tf, dt));
}

BOOST_AUTO_TEST_CASE(spectrum_diagonalization, * utf::tolerance(1e-9)) {
	
}

BOOST_AUTO_TEST_SUITE_END()
