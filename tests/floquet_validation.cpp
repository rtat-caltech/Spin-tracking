#include <boost/test/unit_test.hpp>
#include <math.h>
#include "../include/particle.h"
#include "../include/quaternion.h"
#include "../include/double3.h"
#include "../include/floquet.h"
#include "../include/integrator.h"
#include "../include/simulation.h"
#include "../include/optionsParser.h"
#include <iostream>
#include <stdlib.h>

namespace utf = boost::unit_test;
using namespace std;

namespace boost { namespace math { namespace fpc {
			template <>
			struct tolerance_based< quaternion > : boost::true_type{};
		} } }

bool coords_compare(coords a, coords b, double tol) {
	return len(a - b) < tol;
}

complex<double> sine_spectrum(double a, double w1, double w2, double t0, double tf, double dt) {
	// Computes 2 J(w2)
	// 2 Re[J(w2)] == S(w2)
	double real_part = 0.0;
	double im_part = 0.0;
	double t1 = first_sample_point(t0, dt);
	int n = 0;
	double nsamp = (tf - t0)/dt;
	complex<double> total;
	complex<double> partial_total = 0.0;
	
	double s1 = 0;
	double s2 = 0;
	double angle = w2 * dt;
	while (t1 < first_sample_point(tf, dt) - dt/2) {
		double x = a * sin(w1 * t1);
		complex<double> x2 = x * exp(w2 * t1 * im_unit);
		partial_total += x2;
		total += partial_total * conj(x2) - x * x/2;
		t1 += dt;
		n++;
	}

	return total * 2.0 * dt/(n + 0.0);
}

BOOST_AUTO_TEST_SUITE(FloquetIntegration)

BOOST_AUTO_TEST_CASE(diagonalization, * utf::tolerance(quaternion(1e-9))) {
	quaternion q = {2, 3, -5, 7};
	q = q/norm(q);
	quaternion eigen_values = qEigenval(q);
	quaternion eigen_vectors = qEigenvec(q);
	BOOST_TEST(eigen_values.x == 0);
	BOOST_TEST(eigen_values.y == 0);
	BOOST_TEST(norm(eigen_values) == 1);
	BOOST_TEST(norm(eigen_vectors) == 1);
	
	quaternion a = eigen_vectors * eigen_values * conj(eigen_vectors);
	
	// Make sure matrix is diagonal
	Matrix2cd lambdas = toSU2(eigen_values);
	BOOST_TEST(lambdas(0,1) == complex<double>(0, 0));
	BOOST_TEST(lambdas(1,0) == complex<double>(0, 0));
	
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
	compileOptions(opt);
	coords s0 = {0, 1, 0};
	coords s = {0, 1, 0};

	int n_prop = 100;
	vector<quaternion> propagators;
	coords dummy = {0, 0, 0};
	double h = 1e-6;
	floquetDiagonalization fd = floquet_diagonalize(opt);
	for (int i=0; i < n_prop; i++) {
		double t1 = t0 + (tf - t0) * i/n_prop;
		double t2 = t0 + (tf - t0) * (i+1)/n_prop;
		integrateMagnusCFET(t1, t2, s, dummy, dummy, dummy, dummy, opt, h);
		coords a = fd.propagators[i] * s0;
		coords b = s;
	  	BOOST_TEST(a.x == b.x);
		BOOST_TEST(a.y == b.y);
		BOOST_TEST(a.z == b.z);
	}
	cout << "Floquet:" << endl;
	cout << fd.f_modes_0 << endl;
	cout << fd.f_energies << endl;
	// Look in scripts/qutipFloquet.py for how these tensors are calculated
	double Delta_ref[2][2][NK] = {{{-31415.9265359, -25132.74122872, -18849.55592154, -12566.37061436, -6283.18530718, 0., 6283.18530718, 12566.37061436, 18849.55592154, 25132.74122872, 31415.9265359}, {-30941.87521652, -24658.68990934, -18375.50460216, -12092.31929499, -5809.13398781, 474.05131937, 6757.23662655, 13040.42193373, 19323.60724091, 25606.79254809, 31889.97785527}}, {{-31889.97785527, -25606.79254809, -19323.60724091, -13040.42193373, -6757.23662655, -474.05131937, 5809.13398781, 12092.31929499, 18375.50460216, 24658.68990934, 30941.87521652}, {-31415.9265359, -25132.74122872, -18849.55592154, -12566.37061436, -6283.18530718, 0., 6283.18530718, 12566.37061436, 18849.55592154, 25132.74122872, 31415.9265359}}};

	double X_ref[2][2][NK] = {{{4.82902543e-08, 1.42500174e-29, 3.30960564e-04, 9.12919515e-29, 1.87688610e-01, 1.42211545e-28, 1.87688610e-01, 9.12898629e-29, 3.30960564e-04, 1.42716098e-29, 4.82902543e-08}, {3.15136984e-29, 7.15550842e-06, 1.27178379e-28, 1.57109229e-02, 1.07067023e-26, 5.98999978e-01, 4.95743979e-27, 9.23919626e-03, 2.17816906e-28, 3.50971835e-06, 2.10895130e-29}}, {{2.10986163e-29, 3.50971835e-06, 2.17820263e-28, 9.23919626e-03, 4.95687493e-27, 5.98999978e-01, 1.07096047e-26, 1.57109229e-02, 1.27314158e-28, 7.15550842e-06, 3.12693008e-29}, {4.82902544e-08, 6.83511992e-28, 3.30960564e-04, 3.54274315e-27, 1.87688610e-01, 1.09678335e-26, 1.87688610e-01, 3.54106031e-27, 3.30960564e-04, 6.83481738e-28, 4.82902544e-08}}};
	
	double X_re_ref[2][2][NK] = {{{-5.62375667e-16, 1.12019768e-15, -8.00238781e-15, 3.56463990e-15, -7.13459240e-15, 1.19252481e-14, -7.14847018e-15, 3.56463990e-15, -8.00130361e-15, 1.11965558e-15, -5.61941986e-16}, {-2.64942514e-15, 1.06418386e-14, 5.79204954e-15, 1.36181017e-14, -3.61154889e-14, -4.43572181e-15, 3.46068216e-14, -8.31895517e-15, -3.87040961e-15, -8.54152220e-15, 2.62297060e-15}}, {{2.62470533e-15, -8.54152220e-15, -3.87084329e-15, -8.33977185e-15, 3.45925101e-14, -4.43462405e-15, -3.61742527e-14, 1.36285101e-14, 5.80375892e-15, 1.06435734e-14, -2.64855778e-15}, {-7.43047117e-15, -1.25160299e-14, 2.13183421e-14, -7.20354765e-15, 6.31592218e-14, -1.04727425e-13, 6.31592218e-14, -7.20007821e-15, 2.13189926e-14, -1.25151625e-14, -7.42938697e-15}}};
	
	double X_im_ref[2][2][NK] = {{{2.19750437e-04, -3.60488205e-15, 1.81923216e-02, -8.86483468e-15, 4.33230435e-01, -1.44712250e-18, -4.33230435e-01, 8.86471687e-15, -1.81923216e-02, 3.60804396e-15, -2.19750437e-04}, {4.94916608e-15, -2.67497821e-03, 9.67628755e-15, -1.25343221e-01, -9.69658381e-14, -7.73950888e-01, 6.13172707e-14, -9.61207379e-02, 1.42420797e-14, -1.87342423e-03, 3.76955411e-15}}, {{-3.76955411e-15, 1.87342423e-03, -1.42420797e-14, 9.61207379e-02, -6.13207402e-14, 7.73950888e-01, 9.69588992e-14, 1.25343221e-01, -9.67628755e-15, 2.67497821e-03, -4.92487995e-15}, {-2.19750437e-04, 2.29534526e-14, -1.81923216e-02, 5.90834329e-14, -4.33230435e-01, -2.62499347e-18, 4.33230435e-01, -5.90696130e-14, 1.81923216e-02, -2.29532665e-14, 2.19750437e-04}}};

	double Gamma_ref[2][2][NK] = {{{0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 4.46770744e-28, 2.91336071e-02, 3.60945112e-30, 5.83624178e-06, 1.41737937e-31, 3.07114136e-10}, {0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 3.07304001e+00, 6.67559967e-28, 3.39378535e-04, 3.65538839e-30, 3.35799422e-08, 1.30170073e-31}}, {{0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.93663610e-27, 6.70505594e-04, 2.36207527e-30, 7.38187545e-08, 2.04998836e-31}, {0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 3.44564651e-26, 2.91336071e-02, 1.40007704e-28, 5.83624178e-06, 6.78797223e-30, 3.07114136e-10}}};
	
	double A_ref[2][2] = {{2.91394437e-02, 3.07337942e+00}, {6.70579412e-04, 2.91394437e-02}};
	
	coords b_ref = {0., 0.99502715, 0.};

	double Delta[2][2][NK] = {{{0}}};
	complex<double> X[2][2][NK] = {{{0}}};
	complex<double> Gamma[2][2][NK] = {{{0}}};
	complex<double> Zeta[2][2] = {{0}};
	complex<double> Omicron[2][2] = {{0}};
	
	double ea = -atan2(fd.f_energies.z, fd.f_energies.w)/fd.period;
	double eb = atan2(fd.f_energies.z, fd.f_energies.w)/fd.period;
	double deltaE = (ea - eb);

	quaternion c_op = {0, 0, 1, 0};
	Spectrum spec;
	for(int k=0; k <= NK/2; k++) {
		for (int i=-1; i < 2; i++) {
			double w = deltaE * i + k * opt.w;
			if (w < 0) {
				continue;
			}			
			spec.frequencies[k*3 + i] = w;
			spec.power[k*3 + i] = 1/(pow(w * (tf - t0), 2) + 1);
		}
	}
	CovarianceSpectrum cspec;
	cspec.initialize(spec.frequencies, 1);
	for (int i = 0; i < NW; i++) {
		cspec.variance[i](1, 1) = spec.power[i];
	}
	

	floquet_Delta(fd, Delta);
	floquet_X(fd, c_op,  X);
	
	vector<quaternion> c_ops;
	c_ops.push_back(quaternion(0, 1, 0, 0));
	c_ops.push_back(c_op);
	c_ops.push_back(quaternion(0, 0, 0, 1));
	floquet_master_equation_rates(fd, c_ops, cspec, Zeta, Omicron);

	for (int i=0; i < 2; i++) {
		for (int j=0; j < 2; j++) {
			// The reference A is for T=0. For T=Infinity (our case), we should use A_ref + A_ref^T
			BOOST_TEST(abs(4 * M_PI * Zeta[i][j]) == A_ref[i][j] + A_ref[j][i]);
			for (int k = 0; k < NK; k++) {
				BOOST_TEST(Delta[i][j][k] == Delta_ref[i][j][k]);
				BOOST_TEST(real(X[i][j][k]) - X_re_ref[i][j][k] == 0);
				BOOST_TEST(imag(X[i][j][k]) - X_im_ref[i][j][k] == 0);
			}
		}
	}

	Matrix2cd rho = bloch_to_density(s0);
	rho = integrateFloquetMarkov(t0, (tf - t0) * 10 + t0, rho, Zeta, Omicron);
	coords sf = density_to_bloch(rho);
	BOOST_TEST(sf.x == b_ref.x);
	BOOST_TEST(sf.y == b_ref.y);
	BOOST_TEST(sf.z == b_ref.z);
}

BOOST_AUTO_TEST_CASE(SU2, * utf::tolerance(1e-10)) {
	// Map identity to identity
	quaternion qid = {1, 0, 0, 0};
	Matrix2cd identity = toSU2(qid);
	BOOST_TEST(identity(0, 0) == complex<double>(1, 0));
	BOOST_TEST(identity(0, 1) == complex<double>(0, 0));
	BOOST_TEST(identity(1, 0) == complex<double>(0, 0));
	BOOST_TEST(identity(1, 1) == complex<double>(1, 0));

	// Group isomorphism? What's that?
	quaternion a = {1, 2, -3, 4};
	a = a/norm(a);
	quaternion b = {10, -5, 6, 3};
	b = b/norm(b);
	Matrix2cd c_su2 = toSU2(a * b);
	Matrix2cd ab_su2 = toSU2(a) * toSU2(b);
	Matrix2cd a_conj = toSU2(a).adjoint();
	Matrix2cd a_conj_2 = toSU2(conj(a));
	for (int i = 0; i < 2; i++) {
		for (int j = 0; j < 2; j++) {
			BOOST_TEST(abs(ab_su2(i,j) - c_su2(i, j)) == 0);
			BOOST_TEST(abs(a_conj(i,j) - a_conj_2(i, j)) == 0);
		}
	}

	// Exponentiation
	coords w = {3, -5, 4};
	coords wv = sin(len(w)/2) * w/len(w);
	quaternion wq = {cos(len(w)/2), wv.x, wv.y, wv.z};
	Matrix2cd w_rotation = toSU2(wq);
	Matrix2cd sx, sy, sz;
	sx << 0, 1,
		1, 0;
	sy << 0, (0.0 - 1.0 * im_unit),
		(0.0 + 1.0 * im_unit), 0;
	sz << 1, 0,
		0, -1;
	Matrix2cd w_rotation_2 = ((sx * w.x + sy * w.y + sz * w.z)/2 * -im_unit).exp();
	for (int i = 0; i < 2; i++) {
		for (int j = 0; j < 2; j++) {
			BOOST_TEST(abs(w_rotation(i,j) - w_rotation_2(i, j)) == 0);
		}
	}	
}

BOOST_AUTO_TEST_CASE(axis_rotation, * utf::tolerance(1e-9)) {
	coords bx = {1, 0, 0};
	coords by = {0, 1, 0};
	coords bz = {0, 0, 1};
	double t = 1/sqrt(2);
	quaternion qrotx = {t, t, 0, 0}; // 90 degree rotation about x axis (rotates y into z)
	quaternion qroty = {t, 0, t, 0};
	quaternion qrotz = {t, 0, 0, t};

	coords r_xy = qv_mult(qrotx, by);
	coords r_yz = qv_mult(qroty, bz);
	coords r_zx = qv_mult(qrotz, bx);

	BOOST_TEST(r_xy.x == bz.x);
	BOOST_TEST(r_xy.y == bz.y);
	BOOST_TEST(r_xy.z == bz.z);

	BOOST_TEST(r_yz.x == bx.x);
	BOOST_TEST(r_yz.y == bx.y);
	BOOST_TEST(r_yz.z == bx.z);

	BOOST_TEST(r_zx.x == by.x);
	BOOST_TEST(r_zx.y == by.y);
	BOOST_TEST(r_zx.z == by.z);

	Vector2cd px, py, pz, p_xy, p_yz, p_zx;
	px << t, t;
	py << t, im_unit * t;
	pz << 1, 0;
	p_xy = toSU2(qrotx) * py;
	p_yz = toSU2(qroty) * pz;
	p_zx = toSU2(qrotz) * px;

	cout << p_xy << endl;
	cout << p_yz << endl;
	cout << p_zx << endl;
	BOOST_TEST(abs((complex<double>) (p_xy.adjoint() * pz)) == 1);
	BOOST_TEST(abs((complex<double>) (p_yz.adjoint() * px)) == 1);
	BOOST_TEST(abs((complex<double>) (p_zx.adjoint() * py)) == 1);
}

BOOST_AUTO_TEST_CASE(proper_rotation, * utf::tolerance(1e-9)) {
	quaternion a = {1, 2, -3, 4};
	a = a/norm(a);
	coords b = {-5, 6, 3};

	coords b2 = density_to_bloch(bloch_to_density(b));
	BOOST_TEST(b.x == b2.x);
	BOOST_TEST(b.y == b2.y);
	BOOST_TEST(b.z == b2.z);	

	coords c = qv_mult(a, b);
	coords d = density_to_bloch(toSU2(a) * bloch_to_density(b) * toSU2(a).adjoint());
	coords e = density_to_bloch(bloch_to_density(b, conj(a)));
	BOOST_TEST(c.x == d.x);
	BOOST_TEST(c.y == d.y);
	BOOST_TEST(c.z == d.z);
	BOOST_TEST(c.x == e.x);
	BOOST_TEST(c.y == e.y);
	BOOST_TEST(c.z == e.z);
}

BOOST_AUTO_TEST_CASE(transformations, * utf::tolerance(1e-9)) {
	double t0 = 0.0;
	double tf = 1;
	options opt;
	opt.gravity = false;
	opt.T = 0.25;
	opt.B0 = {5e-6, 0.0, 0.0};
	opt.E = {0, 0, 0};
	opt.a = 3e-5;
	opt.w = 1e3 * 2 * M_PI;
	opt.atol = 1e-13;
	opt.rtol = 1e-13;
	compileOptions(opt);

	coords s0, s1, res;
	floquetDiagonalization fd;
	coords zeros = {0, 0, 0};
	_PREC h;
	s0 = {0, 0, 1};
	s1 = {0, 0, 1};
	h = 1e-6;
	integrateDOP(t0, tf, s0, zeros, zeros, zeros, zeros, opt, h);
	fd = floquet_diagonalize(opt);
	res = density_to_bloch(bloch_to_density(s1, fd.f_modes_0),
	                              fd.f_modes_0 * pow(fd.f_energies, 1000));

	BOOST_TEST(res.x == s0.x);
	BOOST_TEST(res.y == s0.y);
	BOOST_TEST(res.z == s0.z);

	s0 = {0, 1, 0};
	s1 = {0, 1, 0};
	h = 1e-6;
	integrateDOP(t0, tf, s0, zeros, zeros, zeros, zeros, opt, h);
	res = density_to_bloch(bloch_to_density(s1, fd.f_modes_0),
	                       fd.f_modes_0 * pow(fd.f_energies, 1000));
	BOOST_TEST(res.x == s0.x);
	BOOST_TEST(res.y == s0.y);
	BOOST_TEST(res.z == s0.z);

	s0 = {1, 0, 0};
	s1 = {1, 0, 0};
	h = 1e-6;
	integrateDOP(t0, tf, s0, zeros, zeros, zeros, zeros, opt, h);
	fd = floquet_diagonalize(opt);
	res = density_to_bloch(bloch_to_density(s1, fd.f_modes_0),
	                              fd.f_modes_0 * pow(fd.f_energies, 1000));
	BOOST_TEST(res.x == s0.x);
	BOOST_TEST(res.y == s0.y);
	BOOST_TEST(res.z == s0.z);

	s1 = {1, 0, 0};
	density_to_bloch(bloch_to_density(s1, fd.f_modes_0));
}

BOOST_AUTO_TEST_CASE(goertzel, * utf::tolerance(1e-9)) {
	coords w0 = {1.1e2, 5.0e2, -2.3e2}; // Signal frequencies
	coords a = {1.0, 2.0, 3.0}; // Signal amplitudes
	double w = 1.3e2; // frequency to evaluate FT at
	double t0 = 0.3;
	double tf = 1.2;
	double dt = 1e-4;
	double t = first_sample_point(t0, dt);
	int n;
	coords s1 = {0, 0, 0};
	coords s2 = {0, 0, 0};
	coords r_re = {0, 0, 0};
	coords r_im = {0, 0, 0};
	while (t <= first_sample_point(tf, dt) - (dt/2)) {
		coords x = testNoise(t, a, w0);
		goertzel_stage_1(x, s1, s2, w, dt);
		r_re = r_re + x * cos(-w * t);
		r_im = r_im + x * sin(-w * t);
		t += dt;
		n++;
	}
	pair<coords, coords> p = goertzel_stage_2_vector(s1, s2, w, dt);
	BOOST_TEST(p.first.x = r_re.x);
	BOOST_TEST(p.first.y = r_re.x);
	BOOST_TEST(p.first.z = r_re.x);
	BOOST_TEST(p.second.x = r_im.x);
	BOOST_TEST(p.second.y = r_im.y);
	BOOST_TEST(p.second.z = r_im.z);
}

BOOST_AUTO_TEST_CASE(spectrum_calculation, * utf::tolerance(1e-4)) {
	// Tests whether the noise spectrum is calculated correctly.
	// TODO: The accuracy is lower than I'd like. Maybe want to consider fixing for future
	options opt;
	opt.gravity = false;
	opt.T = 0.4;
	opt.a = 1.0;
	opt.w = 1e3 * 2 * M_PI;
	opt.noiseAmplitudes = {1.0, 2.0, 3.0}; // Artificially inject a sine wave B-field
	opt.noiseFrequencies = {1.0e3, 3.5e3, -3.5e3}; // See the testNoise function in integrators.cpp
	double t0 = 0.1;
	double tf = 1.2;
	coords y = (coords){0, 0, 1};
	coords p_old = (coords) {0, 0, 0};
	coords p_new = (coords) {0, 0, 0};
	coords v_old = (coords) {0, 0, 0};
	coords v_new = (coords) {0, 0, 0};
	double B0 = 3e-6;
	opt.B0 = {B0, 0.0, 0.0};
	opt.E = {0.0, 0.0, 0.0};
	opt.tf = tf;
	opt.t0 = t0;
	opt.rtol = 1e-12;
	opt.atol = 1e-12;
	opt.gravity = false;
	opt.numParticles = 1;
	opt.integratorType = 6;
	compileOptions(opt);
	double dt = 1e-5;

	double w[NW] = {0};
	w[0] = opt.noiseFrequencies.x;
	w[1] = opt.noiseFrequencies.y;
	w[2] = opt.noiseFrequencies.z;
	particle p = particle(opt);
	p.initParticles();
	p.getSpectrumAggregators()[0].initialize(w, dt);
	
	int n_steps = integrateSpectrum(t0, tf, p.getSpectrumAggregators()[0], p_old, p_new, v_old, v_new, opt, dt);
	CovarianceSpectrum spec;
	spec.initialize(w, dt);
	p.aggregateSpectrum(spec, opt.numParticles);
	spec.normalize();
	
	BOOST_TEST(spec.variance[0](0, 0).real() == sine_spectrum(opt.gamma/2 *  opt.noiseAmplitudes.x, opt.noiseFrequencies.x, opt.noiseFrequencies.x, t0, tf, dt).real());
	BOOST_TEST(spec.variance[1](1, 1).real() ==  sine_spectrum(opt.gamma/2 * opt.noiseAmplitudes.y, opt.noiseFrequencies.y, opt.noiseFrequencies.y, t0, tf, dt).real());
	BOOST_TEST(spec.variance[2](2, 2).real() ==  sine_spectrum(opt.gamma/2 * opt.noiseAmplitudes.z, opt.noiseFrequencies.z, opt.noiseFrequencies.z, t0, tf, dt).real());
	BOOST_TEST(spec.variance[0](0, 0).imag() ==  sine_spectrum(opt.gamma/2 * opt.noiseAmplitudes.x, opt.noiseFrequencies.x, opt.noiseFrequencies.x, t0, tf, dt).imag());
	BOOST_TEST(spec.variance[1](1, 1).imag() ==  sine_spectrum(opt.gamma/2 * opt.noiseAmplitudes.y, opt.noiseFrequencies.y, opt.noiseFrequencies.y, t0, tf, dt).imag());
	BOOST_TEST(spec.variance[2](2, 2).imag() ==  sine_spectrum(opt.gamma/2 * opt.noiseAmplitudes.z, opt.noiseFrequencies.z, opt.noiseFrequencies.z, t0, tf, dt).imag());
	
}

BOOST_AUTO_TEST_CASE(free_spectrum, * utf::tolerance(1e-6)) {
	// dw = g^2/4 Im[S(w_0')] for noise parallel to dressing field
	// - see Quantum Control of Critically Dressed Spin 1/2 Species + Kramers-Kronig Relations
	options opt;
	opt.gravity = false;
	opt.a = 0;
	opt.w = 1e3 * 2 * M_PI;
	double t0 = 0.0;
	double tf = 1.2;
	double B0 = 3e-6;
	opt.B0 = {B0, 0.0, 0.0};
	opt.E = {0.0, 0.0, 0.0};
	opt.tf = tf;
	opt.t0 = t0;
	opt.rtol = 1e-12;
	opt.atol = 1e-12;
	opt.gravity = false;
	opt.numParticles = 1;
	opt.ioutInt = 1.2;
	opt.yi = {0.0, 1.0, 0.0};
	compileOptions(opt);
	double dt = 1e-5;

	floquetDiagonalization fd = floquet_diagonalize(opt);

	double Szz = 5e-3; // This quantity is gamma^2 S_{Bz, Bz}
	
	CovarianceSpectrum cspec;
	int nsamp = (int) ((tf - t0)/dt);
	cspec.initialize(fd.frequencies, dt, nsamp);
	for (int i = 0; i < NW; i++) {
		if (abs(fd.frequencies[i] - abs(opt.gamma * B0)) < 1e-3) {
			// The 1/4 comes from the fact that the perturbation Hamiltonian is
			// H_int(t) = f(t) \sigma = \gamma B(t)/2 \sigma
			cspec.variance[i](2,2) = Szz/4 * im_unit * (nsamp/dt);
		}
	}
	double duration = opt.tf - opt.t0;
	double w0 = B0 * opt.gamma;
	double w2 = w0 + Szz/4;
	
	//BOOST_TEST(abs((w2 - w0)/w0) < 1e-4); //Perturbation is small
	//BOOST_TEST(abs((w2 - w0) * (tf - t0)) > 1e-3); //But not too small

	coords b_ref = {0.0, cos(duration * w2), -sin(duration * w2)};
	coords b0 = {0.0, cos(duration * w0), -sin(duration * w0)};
	vector<coords> b_arr = floquet_integrate(fd, cspec, opt);
	coords b_end = b_arr[b_arr.size() - 1];
	
	double shift_ref = asin(len(cross(b_ref, b0)));
	double shift_fm = asin(len(cross(b_end, b0)));	
	BOOST_TEST(shift_ref == shift_fm);
}

BOOST_AUTO_TEST_SUITE_END()
