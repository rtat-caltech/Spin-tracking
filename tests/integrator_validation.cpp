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
	coords y;
	coords p_old = (coords) {0, 0, 0};
	coords p_new = (coords) {0, 0, 0};
	coords v_old = (coords) {0, 0, 0};
	coords v_new = (coords) {0, 0, 0};
	options opts;
	double B0 = 3e-6;
	double h0 = 1e-4;
	opts.B0 = {B0, 0.0, 0.0};
	opts.E = {0.0, 0.0, 0.0};
	opts.tf = tf;
	opts.t0 = t0;
	opts.rtol = 1e-12;
	opts.atol = 1e-12;
	opts.gravity = false;
	opts.h = h0;
	y = (coords){0, 0, 1};
	integrateRK45(t0, tf, y, p_old, p_new, v_old, v_new, opts, opts.h);
	BOOST_TEST(y.x == 0);
	BOOST_TEST(y.y == sin(B0*opts.gamma*(tf-t0)));
	BOOST_TEST(y.z == cos(B0*opts.gamma*(tf-t0)));

	opts.h = h0;
	y = (coords) {0, 0, 1};
	integrateDOP(t0, tf, y, p_old, p_new, v_old, v_new, opts, opts.h);
	BOOST_TEST(y.x == 0);
	BOOST_TEST(y.y == sin(B0*opts.gamma*(tf-t0)));
	BOOST_TEST(y.z == cos(B0*opts.gamma*(tf-t0)));

	opts.h = h0;
	y = (coords){0, 0, 1};
	integrateRKF45(t0, tf, y, p_old, p_new, v_old, v_new, opts, opts.h);
	BOOST_TEST(y.x == 0);
	BOOST_TEST(y.y == sin(B0*opts.gamma*(tf-t0)));
	BOOST_TEST(y.z == cos(B0*opts.gamma*(tf-t0)));
	
	opts.h = h0;
	y = (coords){0, 0, 1};
	integrateRK45Quaternion(t0, tf, y, p_old, p_new, v_old, v_new, opts, opts.h);
	BOOST_TEST(y.x == 0);
	BOOST_TEST(y.y == sin(B0*opts.gamma*(tf-t0)));
	BOOST_TEST(y.z == cos(B0*opts.gamma*(tf-t0)));

	opts.h = h0;
	y = (coords){0, 0, 1};
	integrateMagnusCFET(t0, tf, y, p_old, p_new, v_old, v_new, opts, opts.h);
	BOOST_TEST(y.x == 0);
	BOOST_TEST(y.y == sin(B0*opts.gamma*(tf-t0)));
	BOOST_TEST(y.z == cos(B0*opts.gamma*(tf-t0)));
}

BOOST_AUTO_TEST_SUITE_END()
