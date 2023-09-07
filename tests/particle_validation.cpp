#include <boost/test/unit_test.hpp>
#include <math.h>
#include "../include/particle.h"
#include "../include/double3.h"
#include "../include/integrator.h"
#include <iostream>
#include <stdlib.h>

namespace utf = boost::unit_test;
using namespace std;

BOOST_AUTO_TEST_SUITE(ParticleTrajectories)

BOOST_AUTO_TEST_CASE(initialization) {
	double3 y0 = (double3) {0, 0, 1};
	options opts;
	opts.gravity = false;
	opts.T = 0.4;
	opts.m = 1e-26;
	opts.dist = 'M';
	double3 vsum = (double3) {0, 0, 0};
	double vsqsum = 0;
	double k = 1.380649e-23;
	unsigned int ntrial = 100000;
	for (int i=0; i < ntrial; i++) {
		particle p = particle(y0, opts, rand(), rand());
		double3 vp = p.getState().v;
		vsum = vsum + vp;
		vsqsum = pow(len(vp), 2) + vsqsum;
	}
	// Pretty loose test of MB distribution
	BOOST_TEST(abs(vsum.x/ntrial/sqrt(k * opts.T/opts.m)) < 2e-2); // This is around 7 sigma
	BOOST_TEST(abs(vsum.y/ntrial/sqrt(k * opts.T/opts.m)) < 2e-2);
	BOOST_TEST(abs(vsum.z/ntrial/sqrt(k * opts.T/opts.m)) < 2e-2);	
	BOOST_TEST(abs(sqrt(vsqsum/ntrial)/sqrt(3 * k * opts.T/opts.m) - 1) < 2e-2);
	//TODO: test 'C' distribution initialization	
}

BOOST_AUTO_TEST_CASE(interpolation, * utf::tolerance(1e-9)) {
	double t0 = 0.0;
	double t = 1.0;
	double tf = 2.0;
	double3 g = (double3) {0.0, -9.81, 0.0};
	double3 p_old = (double3) {1, 2, 3};
	double3 v_old = (double3) {1, 2, 3};
	double3 p_new = (double3) {3, 6, 9};
	double3 v_new = (double3) {-3, 12, 0};
	double3 p_out = (double3) {0, 0, 0};
	double3 v_out = (double3) {0, 0, 0};

	// No gravity
	interpolate(t, t0, tf, p_old, p_new, v_old, v_new, p_out, v_out);
	BOOST_TEST(p_out.x == 2);
	BOOST_TEST(p_out.y == 4);
	BOOST_TEST(p_out.z == 6);
	BOOST_TEST(v_out.x == 1);
	BOOST_TEST(v_out.y == 2);
	BOOST_TEST(v_out.z == 3);

	p_new = p_old + v_old * (tf - t0) + 0.5 * g * (tf - t0) * (tf - t0);
	double3 p_mid = p_old + v_old * (t - t0) + 0.5 * g * (t - t0) * (t - t0);
	double3 v_mid = v_old + g * (t - t0);
	// With gravity... hasn't been implemented yet?
	interpolate(t, t0, tf, p_old, p_new, v_old, v_new, p_out, v_out);
	BOOST_TEST(p_out.x == p_mid.x);
	BOOST_TEST(p_out.y == p_mid.y);
	BOOST_TEST(p_out.z == p_mid.z);
	BOOST_TEST(v_out.x == v_mid.x);
	BOOST_TEST(v_out.y == v_mid.y);
	BOOST_TEST(v_out.z == v_mid.z);
}

BOOST_AUTO_TEST_SUITE_END()
