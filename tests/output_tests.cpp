#include <boost/test/unit_test.hpp>
#include <math.h>
#include "../include/particle.h"
#include "../include/double3.h"
#include "../include/integrator.h"
#include "../include/outputHandling.h"
#include "../include/simulation.h"
#include "H5Cpp.h"
#include <iostream>
#include <stdlib.h>

namespace utf = boost::unit_test;
using namespace std;

BOOST_AUTO_TEST_SUITE(OutputTesting)

BOOST_AUTO_TEST_CASE(hdf5_output) {
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
	opt.integratorType = 2;
	opt.numParticles = 3;

	int nsave = int((tf - t0)/opt.ioutInt);
	
	char* outputName = "~tmp.hdf5";
	mainAnalysis(opt, 0, "~tmp.hdf5", 1234567);
	/*
	H5File file(outputName, H5F_ACC_RDONLY);
	DataSet time_ds = file.openDataSet("t");
	DataSet spin_ds = file.openDataSet("S");
	Attribute saved_parameters = file.openAttribute("rtol");
	double* rtol_read;
	saved_parameters.read(PredType::NATIVE_DOUBLE, &rtol_read);
	BOOST_TEST(*rtol_read == opt.rtol);
	
	double time_buffer[opt.numParticles][nsave];
	time_ds.read(time_buffer, PredType::NATIVE_DOUBLE);

	CompType mtype(sizeof(coords));
	mtype.insertMember(MEMBER1, HOFFSET(coords, x), PredType::NATIVE_DOUBLE);
	mtype.insertMember(MEMBER2, HOFFSET(coords, y), PredType::NATIVE_DOUBLE);
	mtype.insertMember(MEMBER3, HOFFSET(coords, z), PredType::NATIVE_DOUBLE);

	coords spin_buffer[opt.numParticles][nsave];
	spin_ds.read(spin_buffer, mtype);

	for (int i = 0; i < nsave; i++) {
		for (int n = 0; n < nsave; n++) {
			BOOST_TEST(time_buffer[n][i] == i * opt.ioutInt);
			coords s = spin_buffer[n][i];
			BOOST_TEST(s.y == 0);
			BOOST_TEST(s.y == sin(B0*opt.gamma*(tf-t0)));
			BOOST_TEST(s.z == cos(B0*opt.gamma*(tf-t0)));
		}
	}
	*/
}

BOOST_AUTO_TEST_SUITE_END()
