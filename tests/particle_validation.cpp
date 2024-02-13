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
	coords y0 = (coords) {0, 0, 1};
	options opt;
	double k = 1.380649e-23;
	opt.gravity = false;
	opt.T = 0.4;
	opt.m = 1e-26;
	opt.dist = 'M';
	opt.numParticles = 100000;
	opt.sqrtKT_m = sqrt(k * opt.T/opt.m);
	particle p = particle(opt);
	p.initParticles();
	coords vsum = (coords) {0, 0, 0};
	double vsqsum = 0;
	unsigned int ntrial = 100000;
	coords* velocities = p.getVelocities();
	for (int i=0; i < opt.numParticles; i++) {
		coords vp = velocities[i];
		vsum = vsum + vp;
		vsqsum = pow(len(vp), 2) + vsqsum;
	}
	// Pretty loose test of MB distribution
	BOOST_TEST(abs(vsum.x/ntrial/sqrt(k * opt.T/opt.m)) < 2e-2); // This is around 7 sigma
	BOOST_TEST(abs(vsum.y/ntrial/sqrt(k * opt.T/opt.m)) < 2e-2);
	BOOST_TEST(abs(vsum.z/ntrial/sqrt(k * opt.T/opt.m)) < 2e-2);	
	BOOST_TEST(abs(sqrt(vsqsum/ntrial)/sqrt(3 * k * opt.T/opt.m) - 1) < 2e-2);
	//TODO: test 'C' distribution initialization	
}

BOOST_AUTO_TEST_CASE(interpolation, * utf::tolerance(1e-9)) {
	double t0 = 0.0;
	double t = 1.0;
	double tf = 2.0;
	coords g = (coords) {0.0, -9.81, 0.0};
	coords p_old = (coords) {1, 2, 3};
	coords v_old = (coords) {1, 2, 3};
	coords p_new = (coords) {3, 6, 9};
	coords v_new = (coords) {-3, 12, 0};
	coords p_out = (coords) {0, 0, 0};
	coords v_out = (coords) {0, 0, 0};

	options opt;
	opt.gravity = false;
	// No gravity
	interpolate(t, t0, tf, p_old, p_new, v_old, v_new, p_out, v_out, opt);
	BOOST_TEST(p_out.x == 2);
	BOOST_TEST(p_out.y == 4);
	BOOST_TEST(p_out.z == 6);
	BOOST_TEST(v_out.x == 1);
	BOOST_TEST(v_out.y == 2);
	BOOST_TEST(v_out.z == 3);

	p_new = p_old + v_old * (tf - t0) + 0.5 * g * (tf - t0) * (tf - t0);
	coords p_mid = p_old + v_old * (t - t0) + 0.5 * g * (t - t0) * (t - t0);
	coords v_mid = v_old + g * (t - t0);

	//With gravity
	opt.gravity = true;
	interpolate(t, t0, tf, p_old, p_new, v_old, v_new, p_out, v_out, opt);
	BOOST_TEST(p_out.x == p_mid.x);
	BOOST_TEST(p_out.y == p_mid.y);
	BOOST_TEST(p_out.z == p_mid.z);
	BOOST_TEST(v_out.x == v_mid.x);
	BOOST_TEST(v_out.y == v_mid.y);
	BOOST_TEST(v_out.z == v_mid.z);
}

BOOST_AUTO_TEST_SUITE_END()
