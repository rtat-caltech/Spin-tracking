#define BOOST_TEST_MODULE  integrator_tests 
#include <boost/test/unit_test.hpp>
#include <math.h>
#include "../include/integrator.h"
#include "../include/double3.h"

BOOST_AUTO_TEST_CASE(free_precession, * utf::tolerance(1e-10)) {
	// Free precession test
	double t0 = 0.1;
	double tf = 0.2;
	double3 y = (double3){1, 0, 0};
	double3 p_old = (double3) {0, 0, 0};
	double3 p_new = (double3) {0, 0, 0};
	double3 v_old = (double3) {0, 0, 0};
	double3 v_new = (double3) {0, 0, 0};
	options opts;
	double B0 = 3e-6;
	opts.B0 = {B0, 0.0, 0.0};
	opts.E = {0.0, 0.0, 0.0};
	opt.tf = tf;
	opt.t0 = t0;
	opt.rtol = 1e-12;
	opt.atol = 1e-12;
	opt.gravity = false;
	int nsteps = integrateDOP(t0, tf, y, p_old, p_new, v_old, v_new, OPT);
	BOOST_TEST(y.x == cos(B0*opts.gamma*(tf-t0)));
	BOOST_TEST(y.y == sin(B0*opts.gamma*(tf-t0)));
	BOOST_TEST(y.z == 0);
}
