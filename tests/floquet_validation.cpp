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

bool double3_compare(double3 a, double3 b, double tol) {
	return len(a - b) < tol;
}

double sine_spectrum(double a, double w1, double w2, double t0, double tf, double dt) {	
	double real_part = 0.0;
	double im_part = 0.0;
	double t = first_sample_point(t0, dt);
	int n = 0;
	while (t <= first_sample_point(tf, dt) - (dt/2)) {
		real_part += sin(w1 * t) * cos(w2 * t);
		im_part += sin(w1 * t) * sin(w2 * t);
		t += dt;
		n++;
	}
	return a * a * (real_part * real_part + im_part * im_part) * dt/n;
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
	double3 s0 = {0, 1, 0};
	double3 s = {0, 1, 0};

	int n_prop = 100;
	quaternion* propagators = (quaternion*) malloc(sizeof(quaternion) * n_prop);
	quaternion y = {1, 0, 0, 0};
	double3 dummy = {0, 0, 0};
	double h = 1e-6;
	for (int i=0; i < n_prop; i++) {
		double t1 = t0 + (tf - t0) * i/n_prop;
		double t2 = t0 + (tf - t0) * (i+1)/n_prop;
		integrateHamiltonian(t1, t2, y, opt, 1e-6);
		integrateMagnusCFET(t1, t2, s, dummy, dummy, dummy, dummy, opt, h);
		double3 a = y * s0;
		double3 b = s;
	  	BOOST_TEST(a.x == b.x);
		BOOST_TEST(a.y == b.y);
		BOOST_TEST(a.z == b.z);
		propagators[i] = y;
	}
	// Look in scripts/qutipFloquet.py for how these tensors are calculated
	double Delta_ref[2][2][NK] = {{{-31415.9265359, -25132.74122872, -18849.55592154, -12566.37061436, -6283.18530718, 0., 6283.18530718, 12566.37061436, 18849.55592154, 25132.74122872, 31415.9265359}, {-30941.87521652, -24658.68990934, -18375.50460216, -12092.31929499, -5809.13398781, 474.05131937, 6757.23662655, 13040.42193373, 19323.60724091, 25606.79254809, 31889.97785527}}, {{-31889.97785527, -25606.79254809, -19323.60724091, -13040.42193373, -6757.23662655, -474.05131937, 5809.13398781, 12092.31929499, 18375.50460216, 24658.68990934, 30941.87521652}, {-31415.9265359, -25132.74122872, -18849.55592154, -12566.37061436, -6283.18530718, 0., 6283.18530718, 12566.37061436, 18849.55592154, 25132.74122872, 31415.9265359}}};

	double X_ref[2][2][NK] = {{{4.82902543e-08, 1.42500174e-29, 3.30960564e-04, 9.12919515e-29, 1.87688610e-01, 1.42211545e-28, 1.87688610e-01, 9.12898629e-29, 3.30960564e-04, 1.42716098e-29, 4.82902543e-08}, {3.15136984e-29, 7.15550842e-06, 1.27178379e-28, 1.57109229e-02, 1.07067023e-26, 5.98999978e-01, 4.95743979e-27, 9.23919626e-03, 2.17816906e-28, 3.50971835e-06, 2.10895130e-29}}, {{2.10986163e-29, 3.50971835e-06, 2.17820263e-28, 9.23919626e-03, 4.95687493e-27, 5.98999978e-01, 1.07096047e-26, 1.57109229e-02, 1.27314158e-28, 7.15550842e-06, 3.12693008e-29}, {4.82902544e-08, 6.83511992e-28, 3.30960564e-04, 3.54274315e-27, 1.87688610e-01, 1.09678335e-26, 1.87688610e-01, 3.54106031e-27, 3.30960564e-04, 6.83481738e-28, 4.82902544e-08}}};

	double Gamma_ref[2][2][NK] = {{{0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 4.46770744e-28, 2.91336071e-02, 3.60945112e-30, 5.83624178e-06, 1.41737937e-31, 3.07114136e-10}, {0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 3.07304001e+00, 6.67559967e-28, 3.39378535e-04, 3.65538839e-30, 3.35799422e-08, 1.30170073e-31}}, {{0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.93663610e-27, 6.70505594e-04, 2.36207527e-30, 7.38187545e-08, 2.04998836e-31}, {0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 3.44564651e-26, 2.91336071e-02, 1.40007704e-28, 5.83624178e-06, 6.78797223e-30, 3.07114136e-10}}};
	
	double A_ref[2][2] = {{2.91394437e-02, 3.07337942e+00}, {6.70579412e-04, 2.91394437e-02}};
	
	double3 b_ref = {0., 0.98446036, -0.03025961};

	double Delta[2][2][NK] = {{{0}}};
	double X[2][2][NK] = {{{0}}};
	double Gamma[2][2][NK] = {{{0}}};
	double A[2][2] = {{0}};
	
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
			if (w < 0) {p
				continue;
			}			
			spec.frequencies[k*3 + i] = w;
			spec.power[k*3 + i] = 1/(pow(w * (tf - t0), 2) + 1);
		}
	}
	
	floquet_master_equation_rates(eigen_vectors, eigen_values, c_op, propagators, n_prop, tf - t0, spec, Delta, X, Gamma, A);

	for (int i=0; i < 2; i++) {
		for (int j=0; j < 2; j++) {
			BOOST_TEST(A[i][j] == A_ref[i][j]);
			cout << i << ", " << j << ", " << A[i][j] << endl;
			for (int k = 0; k < NK; k++) {
				BOOST_TEST(Delta[i][j][k] == Delta_ref[i][j][k]);
				BOOST_TEST(X[i][j][k] - X_ref[i][j][k] == 0);
				BOOST_TEST(Gamma[i][j][k] - Gamma_ref[i][j][k] == 0);
			}
		}
	}

	Matrix2cd rho = bloch_to_density(s0);
	cout << "rho:" << endl;
	cout << rho << endl;
	rho = integrateFloquetMarkov(t0, (tf - t0) * 10 + t0, rho, A);
	cout << rho << endl;
	double3 sf = density_to_bloch(rho);
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
	double3 y = (double3){0, 0, 1};
	double3 p_old = (double3) {0, 0, 0};
	double3 p_new = (double3) {0, 0, 0};
	double3 v_old = (double3) {0, 0, 0};
	double3 v_new = (double3) {0, 0, 0};
	double B0 = 3e-6;
	opt.B0 = {B0, 0.0, 0.0};
	opt.E = {0.0, 0.0, 0.0};
	opt.tf = tf;
	opt.t0 = t0;
	opt.rtol = 1e-12;
	opt.atol = 1e-12;
	opt.gravity = false;
	double dt = 1e-5;

	SpectrumAggregator specagg;
	double w[NW] = {0};
	w[0] = opt.noiseFrequencies.x;
	w[1] = opt.noiseFrequencies.y;
	w[2] = opt.noiseFrequencies.z;
	particle* particles = (particle*) malloc(sizeof(particle)); // Just 1 particle
	particles[0] = particle(opt.yi, opt, 0, 0);
	particles[0].specagg.initialize(w, dt);
	
	int n_steps = integrateSpectrum(t0, tf, particles[0].specagg, p_old, p_new, v_old, v_new, opt, dt);
	CovarianceSpectrum spec;
	spec.initialize(w, dt);
	aggregateSpectrum(particles, spec, 1);
	spec.normalize();

	BOOST_TEST(spec.variance[0](0, 0).real() == opt.gamma * opt.gamma * sine_spectrum(opt.noiseAmplitudes.x, opt.noiseFrequencies.x, opt.noiseFrequencies.x, t0, tf, dt));
	BOOST_TEST(spec.variance[1](1, 1).real() == opt.gamma * opt.gamma * sine_spectrum(opt.noiseAmplitudes.y, opt.noiseFrequencies.y, opt.noiseFrequencies.y, t0, tf, dt));
	BOOST_TEST(spec.variance[2](2, 2).real() == opt.gamma * opt.gamma * sine_spectrum(opt.noiseAmplitudes.z, opt.noiseFrequencies.z, opt.noiseFrequencies.z, t0, tf, dt));

	free(particles);
}

BOOST_AUTO_TEST_CASE(spectrum_diagonalization, * utf::tolerance(1e-9)) {
	
}

BOOST_AUTO_TEST_SUITE_END()
