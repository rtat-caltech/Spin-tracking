#include <boost/test/unit_test.hpp>
#include <math.h>
#include "../include/particle.h"
#include "../include/quaternion.h"
#include "../include/double3.h"
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

double sine_spectrum(double a, double w1, double w2, double t0, double tf, double dt) {	
	double real_part = 0.0;
	double im_part = 0.0;
	double t = first_sample_point(t0, dt);
	while (t <= first_sample_point(tf, dt) - (dt/2)) {
		real_part += sin(w1 * t) * cos(w2 * t);
		im_part += sin(w1 * t) * sin(w2 * t);
		t += dt;
	}
	return a * a * (real_part * real_part + im_part * im_part) * dt/n;
}

BOOST_AUTO_TEST_SUITE(FloquetIntegration)

BOOST_AUTO_TEST_CASE(diagonalization, * utf::tolerance(quaternion(1e-9))) {
	quaternion q = {2, 3, -5, 7};
	q = q/norm(q);
	quaternion eigen_values = qEigenval(q);
	quaternion eigen_vectors = qEigenvec(q);
	BOOST_TEST(eigen_values.y == 0);
	BOOST_TEST(eigen_values.z == 0);
	quaternion a = eigen_vectors * eigen_values * conj(eigen_vectors);
	BOOST_TEST(q == a);
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
	opt.h = 1e-5;
	
	double3 s1[NK];
	double3 s2[NK];
	double w[NK];
	double3 power[NK];
	for (int i=0; i < NK; i++) {
		s1[i] = {0.0, 0.0, 0.0};
		s2[i] = {0.0, 0.0, 0.0};
		w[i] = 0.0;
		power[i] = {0.0, 0.0, 0.0};
	}
	w[0] = opt.noiseFrequencies.x;
	w[1] = opt.noiseFrequencies.y;
	w[2] = opt.noiseFrequencies.z;
	

	int n_steps = integrateSpectrum(t0, tf, s1, s2, w, NK, p_old, p_new, v_old, v_new, opt, opt.h);
	particle* particles = (particle*) malloc(sizeof(particle)); // Just 1 particle
	particles[0] = particle(opt.yi, opt, 0, 0);
	memcpy(particles[0].s1, s1, NK * sizeof(double3));
	memcpy(particles[0].s2, s2, NK * sizeof(double3));
	memcpy(particles[0].w, w, NK * sizeof(double));
	spectrum spec;
	memcpy(spec.frequencies, w, NK * sizeof(double));
	memcpy(spec.power, power, NK * sizeof(double3));
	spec.size = NK;
	aggregateSpectrum(particles, spec, 1, n_steps, opt);
	BOOST_TEST(spec.power[0].x == opt.gamma * opt.gamma * sine_spectrum(opt.noiseAmplitudes.x, opt.noiseFrequencies.x, opt.noiseFrequencies.x, t0, tf, opt.h));
	BOOST_TEST(spec.power[1].y == opt.gamma * opt.gamma * sine_spectrum(opt.noiseAmplitudes.y, opt.noiseFrequencies.y, opt.noiseFrequencies.y, t0, tf, opt.h));
	BOOST_TEST(spec.power[2].z == opt.gamma * opt.gamma * sine_spectrum(opt.noiseAmplitudes.z, opt.noiseFrequencies.z, opt.noiseFrequencies.z, t0, tf, opt.h));
}

BOOST_AUTO_TEST_SUITE_END()
