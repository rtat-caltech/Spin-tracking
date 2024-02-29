#include <boost/test/unit_test.hpp>
#include <math.h>
#include "../include/particle.h"
#include "../include/double3.h"
#include "../include/integrator.h"
#include "../include/logger.h"
#include "../include/simulation.h"
#include <iostream>
#include <stdlib.h>
#if USEHDF5
#include "H5Cpp.h"
#endif

namespace utf = boost::unit_test;
using namespace std;

BOOST_AUTO_TEST_SUITE(OutputTesting)

#if USEHDF5
BOOST_AUTO_TEST_CASE(hdf5_output, * utf::tolerance(1e-9)) {
	cout << "Testing File Output" << endl;
	double t0 = 0.0;
	double tf = 0.2;
	coords y = (coords){0, 0, 1};
	options opt;
	double B0 = 3e-6;
	opt.B0 = {B0, 0.0, 0.0};
	opt.E = {0.0, 0.0, 0.0};
	opt.T = 0.4;
	opt.tf = tf;
	opt.t0 = t0;
	opt.rtol = 1e-12;
	opt.atol = 1e-12;
	opt.h = 1e-5;
	opt.yi = {0, 0, 1};
	opt.gravity = false;
	opt.ioutInt = 0.05;
	opt.integratorType = 0;
	opt.numParticles = 3;

	int nsave = 5;
	
	char* outputName = "~tmp0583.hdf5";

	mainAnalysis(opt, 0, outputName, 1234567);

	H5File file(outputName, H5F_ACC_RDONLY);
	DataSet time_ds = file.openDataSet("Time");
	DataSet spin_ds = file.openDataSet("Spin");
	CompType mtype(sizeof(coords));
	mtype.insertMember(MEMBER1, HOFFSET(coords, x), PredType::NATIVE_DOUBLE);
	mtype.insertMember(MEMBER2, HOFFSET(coords, y), PredType::NATIVE_DOUBLE);
	mtype.insertMember(MEMBER3, HOFFSET(coords, z), PredType::NATIVE_DOUBLE);

	
	double rtol_read;
 	file.openAttribute("rtol").read(PredType::NATIVE_DOUBLE, &rtol_read);
	BOOST_TEST(rtol_read == opt.rtol);
	coords B0_read;
	file.openAttribute("B0").read(mtype, &B0_read);
	BOOST_TEST(B0_read.x == opt.B0.x);
	BOOST_TEST(B0_read.y == opt.B0.y);
	BOOST_TEST(B0_read.z == opt.B0.z);		

	double time_buffer[nsave][opt.numParticles];
	time_ds.read(time_buffer, PredType::NATIVE_DOUBLE);

	coords spin_buffer[nsave][opt.numParticles];
	spin_ds.read(spin_buffer, mtype);

	for (int ti = 0; ti < nsave; ti++) {
		for (int n = 0; n < opt.numParticles; n++) {
			BOOST_TEST(time_buffer[ti][n] == ti * opt.ioutInt);
			coords s = spin_buffer[ti][n];
			BOOST_TEST(s.x == 0);
			BOOST_TEST(s.y == sin(B0 * opt.gamma * ti * opt.ioutInt));
			BOOST_TEST(s.z == cos(B0 * opt.gamma * ti * opt.ioutInt));
		}
	}
	file.close();
	remove(outputName);
}
#endif

BOOST_AUTO_TEST_SUITE_END()
