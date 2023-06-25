#include <boost/test/unit_test.hpp>
#include <math.h>
#include "../include/particle.h"
#include "../include/double3.h"
#include <iostream>

namespace utf = boost::unit_test;
using namespace std;

BOOST_AUTO_TEST_SUITE(ParticleTrajectories)

BOOST_AUTO_TEST_CASE(specular, * utf::tolerance(1e-9)) {
	BOOST_TEST(true);
}

BOOST_AUTO_TEST_SUITE_END()
