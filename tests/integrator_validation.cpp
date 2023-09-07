#include <boost/test/unit_test.hpp>
#include <math.h>
#include "../include/integrator.h"
#include "../include/double3.h"
#include <iostream>

namespace utf = boost::unit_test;
using namespace std;

BOOST_AUTO_TEST_SUITE(IntegratorValidation)

BOOST_AUTO_TEST_CASE(free_precession, * utf::tolerance(1e-9)) {
	// Free precession test;
	cout << "Testing free precession. With tol=1e-12 for T=0.1s, error should be < 1e-9" << endl;
    double t0 = 0.1;
	double tf = 1.2;
	double3 y = (double3){0, 0, 1};
	double3 p_old = (double3) {0, 0, 0};
	double3 p_new = (double3) {0, 0, 0};
	double3 v_old = (double3) {0, 0, 0};
	double3 v_new = (double3) {0, 0, 0};
	options opts;
	double B0 = 3e-6;
	opts.B0 = {B0, 0.0, 0.0};
	opts.E = {0.0, 0.0, 0.0};
	opts.tf = tf;
	opts.t0 = t0;
	opts.rtol = 1e-12;
	opts.atol = 1e-12;
	opts.gravity = false;
	int nsteps;
	nsteps = integrateDOP(t0, tf, y, p_old, p_new, v_old, v_new, opts, opts.h);
	BOOST_TEST(y.x == 0);
	BOOST_TEST(y.y == sin(B0*opts.gamma*(tf-t0)));
	BOOST_TEST(y.z == cos(B0*opts.gamma*(tf-t0)));
	cout << "DOP steps taken: " << to_string(nsteps) << endl;
	
	y = (double3){0, 0, 1};
	nsteps = integrateRK45Hybrid(t0, tf, y, p_old, p_new, v_old, v_new, opts, opts.h);
	BOOST_TEST(y.x == 0);
	BOOST_TEST(y.y == sin(B0*opts.gamma*(tf-t0)));
	BOOST_TEST(y.z == cos(B0*opts.gamma*(tf-t0)));
	cout << "RK45 steps taken: " << to_string(nsteps) << endl;
	
	y = (double3){0, 0, 1};
	nsteps = integrateMagnusCFET(t0, tf, y, p_old, p_new, v_old, v_new, opts, opts.h);
	BOOST_TEST(y.x == 0);
	BOOST_TEST(y.y == sin(B0*opts.gamma*(tf-t0)));
	BOOST_TEST(y.z == cos(B0*opts.gamma*(tf-t0)));
	cout << "Magnus steps taken: " << to_string(nsteps) << endl;
}

BOOST_AUTO_TEST_SUITE_END()
