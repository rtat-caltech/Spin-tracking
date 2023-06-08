'''
This file is a bunch of Python-based spin integration routines. 

All of these should be defined so they are interchangeable if possible.
'''

import numpy as np
import numba
import DOP853Coefs as DOPCoefs
import RK45Coefs as RK45Coefs

#@numba.jit
def __DefineRK75109Coefs():
	coef = {
		'A1': 0.0,
		'A2': 4/63,
		'A3': 2/21,
		'A4': 1/7,
		'A5': 7/17,
		'A6': 13/24,
		'A7': 7/9,
		'A8': 91/100,
		'A9': 1,
		'B21': 4/63,
		'B31': 1/42,
		'B32': 1/14,
		'B41': 1/28,
		'B42': 0,
		'B43': 3/28,
		'B51': 12551/19652,
		'B52': 0,
		'B53': -48363/19652,
		'B54': 10976/4913,
		'B61': -36616931/27869184,
		'B62': 0,
		'B63': 2370277/442368,
		'B64': -255519173/63700992,
		'B65': 226798819/445906944,
		'B71': -10401401/7164612,
		'B72': 0,
		'B73': 47383/8748,
		'B74': -4914455/1318761,
		'B75': -1498465/7302393,
		'B76': 2785280/3739203,
		'B81': 181002080831/17500000000,
		'B82': 0,
		'B83': -14827049601/400000000,
		'B84': 23296401527134463/857600000000000,
		'B85': 2937811552328081/949760000000000,
		'B86': -243874470411/69355468750,
		'B87': 2857867601589/3200000000000,
		'B91': -228380759/19257212,
		'B92': 0,
		'B93': 4828803/113948,
		'B94': -331062132205/10932626912,
		'B95': -12727101935/3720174304,
		'B96': 22627205314560/4940625496417,
		'B97': -268403949/461033608,
		'B98': 3600000000000/19176750553961,
		'C1': 95/2366,
		'C2': 0,
		'C3': 0,
		'C4': 3822231133/16579123200,
		'C5': 555164087/2298419200,
		'C6': 1279328256/9538891505,
		'C7': 5963949/25894400,
		'C8': 50000000000/599799373173,
		'C9': 28487/712800,
		'CH1': 1689248233/50104356120,
		'CH2': 0,
		'CH3': 0,
		'CH4': 1/4,
		'CH5': 28320758959727/152103780259200,
		'CH6': 66180849792/341834007515,
		'CH7': 31163653341/152322513280,
		'CH8': 36241511875000/394222326561063,
		'CH9': 28487/712800
	}
	return coef

__RK75109Coefs = __DefineRK75109Coefs()

@numba.jit
def calcGamma(v):
	return 1.0/np.sqrt(1-(v/299792458.0)**2.0)

@numba.jit
def vecLen(a):
	return np.sqrt(np.sum(np.square(a)))

@numba.jit
def vecNorm(a):
	return a/vecLen(a)

@numba.jit
def determineMatrix(s):
	return np.array([[0, -s[2], s[1]], 
					 [s[2], 0, -s[0]], 
					 [-s[1], s[0], 0]])

@numba.jit
def determineQuaternion(a, b):
	return np.array([a, b[0], b[1], b[2]])

@numba.jit
def rodriguez(k, dt):
	angle = vecLen(k)
	norm = k[:]/angle
	K = determineMatrix(norm) #calculate the matrix from this vector
	h = angle*dt
	return np.identity(3)+np.sin(h)*K+(1-np.cos(h))*np.dot(K, K)

@numba.jit
def rodriguezQuat(k, dt):
	angle = vecLen(k)
	norm = k/angle
	h = angle * dt
	return determineQuaternion(np.cos(h/2.0), np.sin(h/2.0)*norm)

@numba.jit
def qConjugate(q):
	w, x, y, z = q
	return np.array([w, -x, -y, -z], dtype=q.dtype)

@numba.jit
def qMult(q1, q2):
	w1, x1, y1, z1 = q1
	w2, x2, y2, z2 = q2
	w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
	x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
	y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
	z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
	return np.array([w, x, y, z], dtype=q1.dtype)

@numba.jit
def qv_mult(q1, v1):
	q2 = np.array([0.0, *v1], dtype=q1.dtype)
	return qMult(qMult(q1, q2), qConjugate(q1))[1:]

@numba.jit
def sign(a, b):
	if b < 0.0:
		return -abs(a)
	return abs(a)

@numba.jit
def max_d3(a, b):
	out = []
	for i in range(len(a)):
		out.append(max(a[i], b[i]))
	return np.array(out)

@numba.jit
def interpolate(t, t0, tf, p_old, p_new, v_old, v_new):
	p_out = (p_old*(tf-t) + p_new*(t-t0))/(tf-t0)
	v_out = v_old
	return p_out, v_out

@numba.jit
def findCrossTerm(t, y, BField, EField, consts, t0, tf, p_old, p_new, v_old, v_new, gamma = -1.83247171E8):
	p, v = interpolate(t,t0,tf,p_old,p_new,v_old,v_new)
	E = EField(t, p)
	B = BField(t, p) + 1.0/89875517873681764.0*np.cross(v, E)
	return gamma * B

@numba.jit
def Bloch(t, y, BField, EField, consts, t0, tf, p_old, p_new, v_old, v_new):
	return np.cross(findCrossTerm(t, y, BField, EField, consts, t0, tf, p_old, p_new, v_old, v_new), y)

@numba.jit
def appCross(y, cross, dt):
	rotation = rodriguez(cross, dt)
	return np.dot(rotation, y)

@numba.jit
def BlochNew(t, dt, y, B0, t0, tf , p_old, p_new, v_old, v_new, gamma = -1.83247171E8):
	crossVal = findCrossTerm(t, y, B0, gamma, t0, tf , p_old, p_new, v_old, v_new)
	rotation = rodriguez(crossVal, dt)
	#print(rotation)
	trueS = np.dot(rotation, y)
	return trueS

@numba.jit
def DOP853(particle, intOPT, physOPT, BField, EField):
	t0 = particle['t_old']
	tf = particle['t']
	p_old = particle['x_old']
	p_new = particle['x']
	v_old = particle['v_old']
	v_new = particle['v']
	y = particle['s'][:]
	#COEF = __DOP853Coefs
	yy1 = np.zeros(3)
	k1 = np.zeros(3)
	k2 = np.zeros(3)
	k3 = np.zeros(3)
	k4 = np.zeros(3)
	k5 = np.zeros(3)
	k6 = np.zeros(3)
	k7 = np.zeros(3)
	k8 = np.zeros(3)
	k9 = np.zeros(3)
	k10 = np.zeros(3)
	idid = 0
	iasti = 0
	iord = 0
	reject = 0
	last = 0
	nonsti = 0
	facold = 1.0E-4
	expo1 = 1.0/8.0 - np.float64(intOPT['beta']) * 0.2;
	fac = 0.0
	facc1 = 1.0 / np.float64(intOPT['fac1']);
	facc2 = 1.0 / np.float64(intOPT['fac2']);
	fac11 = 0.0
	posneg = sign(1.0, tf-t0);
	xph = 0.0
	stnum  = 0.0
	stden = 0.0
	err2 = 0.0
	deno = 0.0
	erri = np.zeros(3)
	sqr = np.zeros(3)
	sk = np.zeros(3)
	atoli = np.float64(intOPT['atol'])
	rtoli = np.float64(intOPT['rtol'])
	hlamb = 0.0
	err = 0.0
	hnew = 0.0
	ydiff = np.zeros(3)
	bspl = np.zeros(3)
	nfcn = 0
	nstep = 0
	naccpt = 0 
	nrejct = 0
	hout = 0.0 
	xold = 0.0
	xout = 0.0
	x = t0
	xf = tf
	h = particle['last_spin_step_size']
	i = 0
	n = 3
	last  = 0
	hlamb = 0.0
	iasti = 0
	k1 = Bloch(x, y, BField, EField, physOPT, t0, tf, p_old, p_new, v_old, v_new);
	hmax = np.abs(np.float64(intOPT['hmax']))
	iord = 8
	nfcn += 2
	reject = 0
	xold = x
	outh = h
	while 1:
		outh = h
		if nstep > np.float64(intOPT['nmax']):
			xout = x
			hout = h
			particle['s'] = y[:]
			particle['n_spin_steps'] += nstep
			particle['last_spin_step_size'] = outh
			return particle
		if 0.1 * abs(h) <= abs(x) * np.float64(intOPT['uround']):
			xout = x
			hout = h;
			particle['s'] = y[:]
			particle['n_spin_steps'] += nstep
			particle['last_spin_step_size'] = outh
			return particle
		if (x + 1.01*h - xf) * posneg > 0.0 :
			h = xf - x
			last = 1
		nstep+=1
		yy1 = y + h * DOPCoefs.a21 * k1;
		k2 = Bloch(x+DOPCoefs.c2*h, yy1, BField, EField, physOPT, t0, tf, p_old, p_new, v_old, v_new);

		yy1 = y + h * (DOPCoefs.a31*k1 + DOPCoefs.a32*k2);
		k3 = Bloch(x+DOPCoefs.c3*h, yy1, BField, EField, physOPT, t0, tf, p_old, p_new, v_old, v_new);

		yy1 = y + h * (DOPCoefs.a41*k1 + DOPCoefs.a43*k3);
		k4 = Bloch(x+DOPCoefs.c4*h, yy1, BField, EField, physOPT, t0, tf, p_old, p_new, v_old, v_new);

		yy1 = y + h * (DOPCoefs.a51*k1 + DOPCoefs.a53*k3 + DOPCoefs.a54*k4);
		k5 = Bloch(x+DOPCoefs.c5*h, yy1, BField, EField, physOPT, t0, tf, p_old, p_new, v_old, v_new);

		yy1 = y + h * (DOPCoefs.a61*k1 + DOPCoefs.a64*k4 + DOPCoefs.a65*k5);
		k6 = Bloch(x+DOPCoefs.c6*h, yy1, BField, EField, physOPT, t0, tf, p_old, p_new, v_old, v_new);

		yy1 = y + h * (DOPCoefs.a71*k1 + DOPCoefs.a74*k4 + DOPCoefs.a75*k5 + DOPCoefs.a76*k6);
		k7 = Bloch(x+DOPCoefs.c7*h, yy1, BField, EField, physOPT, t0, tf, p_old, p_new, v_old, v_new);

		yy1 = y + h * (DOPCoefs.a81*k1 + DOPCoefs.a84*k4 + DOPCoefs.a85*k5 + DOPCoefs.a86*k6 + DOPCoefs.a87*k7);
		k8 = Bloch(x+DOPCoefs.c8*h, yy1, BField, EField, physOPT, t0, tf, p_old, p_new, v_old, v_new);

		yy1 = y + h * (DOPCoefs.a91*k1 + DOPCoefs.a94*k4 + DOPCoefs.a95*k5 + DOPCoefs.a96*k6 + DOPCoefs.a97*k7 + DOPCoefs.a98*k8);
		k9 = Bloch(x+DOPCoefs.c9*h, yy1, BField, EField, physOPT, t0, tf, p_old, p_new, v_old, v_new);

		yy1 = y + h * (DOPCoefs.a101*k1 + DOPCoefs.a104*k4 + DOPCoefs.a105*k5 + DOPCoefs.a106*k6 + DOPCoefs.a107*k7 + DOPCoefs.a108*k8 + DOPCoefs.a109*k9);
		k10 = Bloch(x+DOPCoefs.c10*h, yy1, BField, EField, physOPT, t0, tf, p_old, p_new, v_old, v_new);

		yy1 = y + h * (DOPCoefs.a111*k1 + DOPCoefs.a114*k4 + DOPCoefs.a115*k5 + DOPCoefs.a116*k6 + DOPCoefs.a117*k7 + DOPCoefs.a118*k8 + DOPCoefs.a119*k9 + DOPCoefs.a1110*k10);

		k2 = Bloch(x+DOPCoefs.c11*h, yy1, BField, EField, physOPT, t0, tf, p_old, p_new, v_old, v_new);
		xph = x + h;

		yy1 = y + h * (DOPCoefs.a121*k1 + DOPCoefs.a124*k4 + DOPCoefs.a125*k5 + DOPCoefs.a126*k6 + DOPCoefs.a127*k7 + DOPCoefs.a128*k8 + DOPCoefs.a129*k9 + DOPCoefs.a1210*k10 + DOPCoefs.a1211*k2);

		k3 = Bloch(xph, yy1, BField, EField, physOPT, t0, tf, p_old, p_new, v_old, v_new);
		nfcn += 11;

		k4 = DOPCoefs.b1*k1 + DOPCoefs.b6*k6 + DOPCoefs.b7*k7 + DOPCoefs.b8*k8 + DOPCoefs.b9*k9 + DOPCoefs.b10*k10 + DOPCoefs.b11*k2 + DOPCoefs.b12*k3;
		k5 = y + h * k4;

		# error estimation 
		err = 0.0;
		err2 = 0.0;
		sk = atoli + rtoli * max_d3(np.abs(y), np.abs(k5));
		erri = k4 - np.float64(DOPCoefs.bhh1)*k1 - np.float64(DOPCoefs.bhh2)*k9 - np.float64(DOPCoefs.bhh3)*k3;
		sqr = erri / sk;
		err2 += sum(sqr*sqr);
		erri = np.float64(DOPCoefs.er1)*k1 + np.float64(DOPCoefs.er6)*k6 + np.float64(DOPCoefs.er7)*k7 + np.float64(DOPCoefs.er8)*k8 +\
			np.float64(DOPCoefs.er9)*k9 + np.float64(DOPCoefs.er10)*k10 + np.float64(DOPCoefs.er11)*k2 + np.float64(DOPCoefs.er12)*k3;
		sqr = erri / sk;
		err += sum(sqr*sqr);
		deno = err + 0.01 * err2;
		if deno <= 0.0:
			deno = 1.0;
		err = abs(h) * err * np.sqrt (1.0 / (deno*n));
		# computation of hnew 
		fac11 = pow (err, expo1);
		# Lund-stabilization 
		fac = fac11 / pow(facold,np.float64(intOPT['beta']));
		# we require fac1 <= hnew/h <= fac2 
		fac = max (facc2, min (facc1, fac/np.float64(intOPT['safe'])));
		hnew = h / fac;
		if err <= 1.0:
			# step accepted 
			naccpt+=1;
			k4 = Bloch(xph, k5, BField, EField, physOPT, t0, tf, p_old, p_new, v_old, v_new);
			nfcn+=1;
			# final preparation for dense output 
			k1 = k4;
			y = vecNorm(k5);
			xold = x;
			x = xph;
			if last:
				hout=hnew;
				particle['s'] = y[:]
				particle['n_spin_steps'] += nstep
				particle['last_spin_step_size'] = outh
				return particle
			if abs(hnew) > hmax:
				hnew = posneg * hmax;
			if reject:
				hnew = posneg * min (abs(hnew), abs(h));
			reject = 0;
		else:
			# step rejected
			hnew = h / min (facc1, fac11/np.float64(intOPT['safe']));
			reject = 1;
			if naccpt >= 1:
				nrejct=nrejct + 1;
			last = 0;
		h = hnew;
	particle['s'] = x
	particle['n_spin_steps'] += nstep
	particle['last_spin_step_size'] = outh
	return particle

@numba.jit
def DOP853Rotation(particle, intOPT, physOPT, BField, EField):
	t0 = particle['t_old']
	tf = particle['t']
	p_old = particle['x_old']
	p_new = particle['x']
	v_old = particle['v_old']
	v_new = particle['v']
	y = particle['s'][:]
	COEF = __DOP853Coefs
	idid = 0
	iasti = 0
	iord = 0
	reject = 0
	last = 0
	nonsti = 0
	facold = 1.0E-4
	expo1 = 1.0/8.0 - np.float64(intOPT['beta']) * 0.2;
	fac = 0.0
	facc1 = 1.0 / np.float64(intOPT['fac1']);
	facc2 = 1.0 / np.float64(intOPT['fac2']);
	fac11 = 0.0
	posneg = sign(1.0, tf-t0);
	xph = 0.0
	stnum  = 0.0
	stden = 0.0
	err2 = 0.0
	deno = 0.0
	erri = np.zeros(3)
	sqr = np.zeros(3)
	sk = np.zeros(3)
	atoli = np.float64(intOPT['atol'])
	rtoli = np.float64(intOPT['rtol'])
	hlamb = 0.0
	err = 0.0
	hnew = 0.0
	ydiff = np.zeros(3)
	bspl = np.zeros(3)
	nfcn = 0
	nstep = 0
	naccpt = 0 
	nrejct = 0
	hout = 0.0 
	xold = 0.0
	xout = 0.0
	x = t0
	xf = tf
	h = particle['last_spin_step_size']
	outh = h
	i = 0
	n = 3
	last  = 0
	hlamb = 0.0
	iasti = 0

	hmax = np.abs(np.float64(intOPT['hmax']))
	iord = 8
	nfcn += 2
	reject = 0
	xold = x
	while 1:
		nstep+=1
		outh = h
		if nstep > np.float64(intOPT['nmax']):
			xout = x
			hout = h
			particle['s'] = y[:]
			particle['n_spin_steps'] += nstep
			return particle
		if 0.1 * abs(h) <= abs(x) * np.float64(intOPT['uround']):
			xout = x
			hout = h;
			particle['s'] = y[:]
			particle['n_spin_steps'] += nstep
			return particle
		if (x + 1.01*h - xf) * posneg > 0.0 :
			h = xf - x
			last = 1
		k1q = findCrossTerm(x, y, BField, EField, physOPT, t0, tf, p_old, p_new, v_old, v_new)
		k2q = findCrossTerm(x+COEF['c2']*h, appCross(
			y, k1q, COEF['a21']*h), BField, EField, physOPT, t0, tf, p_old, p_new, v_old, v_new)
		k3q = findCrossTerm(x+COEF['c3']*h, appCross(appCross(
			y, k1q, COEF['a31']*h), k2q, COEF['a32']*h), BField, EField, physOPT, t0, tf, p_old, p_new, v_old, v_new)
		
		k4q = findCrossTerm(x+COEF['c4']*h, appCross(appCross(
			y, k1q, COEF['a41']*h), k3q, COEF['a43']*h), BField, EField, physOPT, t0, tf, p_old, p_new, v_old, v_new)
		
		k5q = findCrossTerm(x+COEF['c5']*h, appCross(appCross(appCross(
			y, k1q, COEF['a51']*h), k3q, COEF['a53']*h), k4q, COEF['a54']*h), BField, EField, physOPT, t0, tf, p_old, p_new, v_old, v_new)
		k6q = findCrossTerm(x+COEF['c6']*h, appCross(appCross(appCross(
			y, k1q, COEF['a61']*h), k4q, COEF['a64']*h), k5q, COEF['a65']*h), BField, EField, physOPT, t0, tf, p_old, p_new, v_old, v_new)
		k7q = findCrossTerm(x+COEF['c7']*h, appCross(appCross(appCross(appCross(
			y, k1q, COEF['a71']*h), k4q, COEF['a74']*h), k5q, COEF['a75']*h), k6q, COEF['a76']*h), BField, EField, physOPT, t0, tf, p_old, p_new, v_old, v_new)
		#print(appCross(appCross(appCross(appCross(appCross(
		#	y, k1q, COEF['a81']*h), k4q, COEF['a84']*h), k5q, COEF['a85']*h), k6q, COEF['a86']*h), k7q, COEF['a87']*h))
		k8q = findCrossTerm(x+COEF['c8']*h, appCross(appCross(appCross(appCross(appCross(
			y, k1q, COEF['a81']*h), k4q, COEF['a84']*h), k5q, COEF['a85']*h), k6q, COEF['a86']*h), k7q, COEF['a87']*h), BField, EField, physOPT, t0, tf, p_old, p_new, v_old, v_new)
		#print(appCross(appCross(appCross(appCross(appCross(appCross(
		#	y, k1q, COEF['a91']*h), k4q, COEF['a94']*h), k5q, COEF['a95']*h), k6q, COEF['a96']*h), k7q, COEF['a97']*h), k8q, COEF['a98']*h))
		k9q = findCrossTerm(x+COEF['c9']*h, appCross(appCross(appCross(appCross(appCross(appCross(
			y, k1q, COEF['a91']*h), k4q, COEF['a94']*h), k5q, COEF['a95']*h), k6q, COEF['a96']*h), k7q, COEF['a97']*h), k8q, COEF['a98']*h), BField, EField, physOPT, t0, tf, p_old, p_new, v_old, v_new)
		#print(appCross(appCross(appCross(appCross(appCross(appCross(appCross(
		#	y, k1q, COEF['a101']*h), k4q, COEF['a104']*h), k5q, COEF['a105']*h), k6q, COEF['a106']*h), k7q, COEF['a107']*h), k8q, COEF['a108']*h), k9q, COEF['a109']*h))
		k10q = findCrossTerm(x+COEF['c10']*h, appCross(appCross(appCross(appCross(appCross(appCross(appCross(
			y, k1q, COEF['a101']*h), k4q, COEF['a104']*h), k5q, COEF['a105']*h), k6q, COEF['a106']*h), k7q, COEF['a107']*h), k8q, COEF['a108']*h), k9q, COEF['a109']*h), BField, EField, physOPT, t0, tf, p_old, p_new, v_old, v_new)
		
		yy1q = appCross(appCross(appCross(appCross(appCross(appCross(appCross(appCross(
			y, k1q, COEF['a111']*h), k4q, COEF['a114']*h), k5q, COEF['a115']*h), k6q, COEF['a116']*h), k7q, COEF['a117']*h), k8q, COEF['a118']*h), k9q, COEF['a119']*h), k10q, COEF['a1110']*h)
		k2q = findCrossTerm(x+COEF['c11']*h, yy1q, BField, EField, physOPT, t0, tf, p_old, p_new, v_old, v_new);
		xph = x + h;
		yy1q = appCross(appCross(appCross(appCross(appCross(appCross(appCross(appCross(appCross(
            y, k1q, h*COEF['a121']), k4q, h*COEF['a124']), k5q, h*COEF['a125']), k6q, h*COEF['a126']), k7q, h*COEF['a127']), k8q, h*COEF['a128']), k9q, h*COEF['a129']), k10q, h*COEF['a1210']), k2q, h*COEF['a1211'])
		#print(yy1q)
		k3q = findCrossTerm(xph, yy1q, BField, EField, physOPT, t0, tf, p_old, p_new, v_old, v_new);
		nfcn += 11;
		k4q = rodriguez(k3q, h*COEF['b12'])@(rodriguez(k2q, h*COEF['b11'])@(rodriguez(k10q, h*COEF['b10'])@(
			rodriguez(k9q, h*COEF['b9'])@(rodriguez(k8q, h*COEF['b8'])@(rodriguez(k7q, h*COEF['b7'])@(rodriguez(k6q, h*COEF['b6'])@rodriguez(k1q, h*COEF['b1'])))))))
		k5q = k4q@y #k4 is the rotation matrix
		# error estimation 
		err = 0.0;
		err2 = 0.0;
		sk = atoli + rtoli * max_d3(np.abs(y), np.abs(k5q));
		#erri = k4 - COEF['bhh1']*k1 - COEF['bhh2']*k9 - COEF['bhh3']*k3
		#This one is a different method of finding the endpoint based on previously calculated values
		erri = k4q@y - appCross(appCross(appCross(y, k1q, COEF['bhh1']*h), k9q, COEF['bhh2']*h), k3q, COEF['bhh3']*h)
		#erri = k4@y - (rodriguez(k1, COEF['bhh1']*h)@(rodriguez(k9, COEF['bhh2']*h)@(rodriguez(k3, COEF['bhh3']*h))))@y;
		#print(erri)
		erri = np.max(np.square(erri))
		sqr = erri / sk;
		err2 += sum(sqr*sqr);
		#erri = COEF['er1']*k1 + COEF['er6']*k6 + COEF['er7']*k7 + COEF['er8']*k8 + COEF['er9']*k9 + COEF['er10']*k10 + COEF['er11']*k2 + COEF['er12']*k3;
		#erri = (np.identity(3) - rodriguez(k1q, COEF['er1']*h)@(rodriguez(k6q, COEF['er6']*h)@(rodriguez(k7q, COEF['er7']*h)@(\
		#	rodriguez(k8q, COEF['er8']*h)@(rodriguez(k9q, COEF['er9']*h)@(rodriguez(k10q, COEF['er10']*h)@(rodriguez(k2q, COEF['er11']*h)@(rodriguez(k3q, COEF['er12']*h)))))))))@y
		erri = appCross(appCross(appCross(appCross(appCross(appCross(appCross(appCross(
			y, k3q, COEF['er12']*h), k2q, COEF['er11']*h), k10q, COEF['er10']*h), k9q, COEF['er9']*h), k8q, COEF['er8']*h), 
										  k7q, COEF['er7']*h), k6q, COEF['er6']*h), k1q, COEF['er1']*h)
		erri = y-erri
		#print(erri)
		erri = np.max(np.square(erri));
		#print(erri)
		sqr = erri / sk;
		err += sum(sqr*sqr);
		deno = err + 0.01 * err2;
		if deno <= 0.0:
			deno = 1.0;
		err = abs(h) * err * np.sqrt (1.0 / (deno*n));
		# computation of hnew 
		fac11 = pow (err, expo1);
		# Lund-stabilization 
		fac = fac11 / pow(facold,np.float64(intOPT['beta']));
		# we require fac1 <= hnew/h <= fac2 
		fac = max (facc2, min (facc1, fac/np.float64(intOPT['safe'])));
		hnew = h / fac;
		if err <= 1.0:
			# step accepted 
			facold = max (err, 1.0E-4);
			naccpt+=1;
			y = k5q[:]
			nfcn+=1;
			# final preparation for dense output 
			xold = x;
			x = xph;
			if last:
				hout=hnew;
				xout = x;
				particle['s'] = y[:]
				particle['n_spin_steps'] += nstep
				particle['last_spin_step_size'] = outh
				return particle
		else:
			# step rejected
			hnew = h / min (facc1, fac11/np.float64(intOPT['safe']));
			reject = 1;
			if naccpt >= 1:
				nrejct=nrejct + 1;
			last = 0;
		h = hnew;
	particle['s'] = y[:]
	particle['n_spin_steps'] += nstep
	particle['last_spin_step_size'] = outh
	return particle

@numba.jit
def DOP853Quaternion(particle, intOPT, physOPT, BField, EField):
	t0 = particle['t_old']
	tf = particle['t']
	p_old = particle['x_old']
	p_new = particle['x']
	v_old = particle['v_old']
	v_new = particle['v']
	y = particle['s'][:]
	COEF = __DOP853Coefs
	idid = 0
	iasti = 0
	iord = 0
	reject = 0
	last = 0
	nonsti = 0
	facold = 1.0E-4
	expo1 = 1.0/8.0 - np.float64(intOPT['beta']) * 0.2;
	fac = 0.0
	facc1 = 1.0 / np.float64(intOPT['fac1']);
	facc2 = 1.0 / np.float64(intOPT['fac2']);
	fac11 = 0.0
	posneg = sign(1.0, tf-t0);
	xph = 0.0
	stnum  = 0.0
	stden = 0.0
	err2 = 0.0
	deno = 0.0
	erri = np.zeros(3)
	sqr = np.zeros(3)
	sk = np.zeros(3)
	atoli = np.float64(intOPT['atol'])
	rtoli = np.float64(intOPT['rtol'])
	hlamb = 0.0
	err = 0.0
	hnew = 0.0
	ydiff = np.zeros(3)
	bspl = np.zeros(3)
	nfcn = 0
	nstep = 0
	naccpt = 0 
	nrejct = 0
	hout = 0.0 
	xold = 0.0
	xout = 0.0
	x = t0
	xf = tf
	h = particle['last_spin_step_size']
	outh = h
	i = 0
	n = 3
	last  = 0
	hlamb = 0.0
	iasti = 0

	hmax = np.abs(np.float64(intOPT['hmax']))
	iord = 8
	nfcn += 2
	reject = 0
	xold = x
	while 1:
		nstep+=1
		outh = h
		if nstep > np.float64(intOPT['nmax']):
			xout = x
			hout = h
			particle['s'] = y[:]
			particle['n_spin_steps'] += nstep
			particle['last_spin_step_size'] = outh
			return particle
		if 0.1 * abs(h) <= abs(x) * np.float64(intOPT['uround']):
			xout = x
			hout = h;
			particle['s'] = y[:]
			particle['n_spin_steps'] += nstep
			particle['last_spin_step_size'] = outh
			return particle
		if (x + 1.01*h - xf) * posneg > 0.0 :
			h = xf - x
			last = 1
		k1q = findCrossTerm(x, y, BField, EField, physOPT, t0, tf, p_old, p_new, v_old, v_new)
		
		temp = qv_mult(rodriguezQuat(k1q, COEF['a21']*h), y)
		
		k2q = findCrossTerm(x+COEF['c2']*h, temp, BField, EField, physOPT, t0, tf, p_old, p_new, v_old, v_new)
		
		temp = qv_mult(qMult(rodriguezQuat(k2q, COEF['a32']*h), rodriguezQuat(k1q, COEF['a31'])), y)
		
		k3q = findCrossTerm(x+COEF['c3']*h, temp, BField, EField, physOPT, t0, tf, p_old, p_new, v_old, v_new)
		
		temp = qv_mult(qMult(rodriguezQuat(k3q, COEF['a43']*h), rodriguezQuat(k1q, COEF['a41']*h)), y)
		
		k4q = findCrossTerm(x+COEF['c4']*h, temp, BField, EField, physOPT, t0, tf, p_old, p_new, v_old, v_new)
		
		temp = qv_mult(qMult(qMult(rodriguezQuat(k4q, COEF['a54']*h), rodriguezQuat(k3q, COEF['a53'])), rodriguezQuat(k1q, COEF['a51']*h)), y)
		
		k5q = findCrossTerm(x+COEF['c5']*h, temp, BField, EField, physOPT, t0, tf, p_old, p_new, v_old, v_new)
		
		temp = qv_mult(qMult(qMult(rodriguezQuat(k5q, COEF['a65']*h), rodriguezQuat(k4q, COEF['a64'])), rodriguezQuat(k1q, COEF['a61']*h)), y)
		
		k6q = findCrossTerm(x+COEF['c6']*h, temp, BField, EField, physOPT, t0, tf, p_old, p_new, v_old, v_new)
		
		temp = qv_mult(qMult(qMult(qMult(rodriguezQuat(k6q, COEF['a76']*h), rodriguezQuat(k5q, COEF['a75'])), 
								   rodriguezQuat(k4q, COEF['a74']*h)), rodriguezQuat(k1q, COEF['a71'])), y)
		
		k7q = findCrossTerm(x+COEF['c7']*h, temp, BField, EField, physOPT, t0, tf, p_old, p_new, v_old, v_new)
		
		temp = qv_mult(qMult(qMult(qMult(qMult(rodriguezQuat(k7q, COEF['a87']*h), rodriguezQuat(k6q, COEF['a86']*h)), 
								   rodriguezQuat(k5q, COEF['a85']*h)), rodriguezQuat(k4q, COEF['a84']*h)), rodriguezQuat(k1q, COEF['a81']*h)), y)
		
		k8q = findCrossTerm(x+COEF['c8']*h, temp, BField, EField, physOPT, t0, tf, p_old, p_new, v_old, v_new)
		
		temp = qv_mult(qMult(qMult(qMult(qMult(qMult(rodriguezQuat(k8q, COEF['a98']*h), rodriguezQuat(k7q, COEF['a97']*h)), 
								   rodriguezQuat(k6q, COEF['a96']*h)), rodriguezQuat(k5q, COEF['a95']*h)), rodriguezQuat(k4q, COEF['a94']*h)), 
								   rodriguezQuat(k1q, COEF['a91']*h)), y)
		
		k9q = findCrossTerm(x+COEF['c9']*h, temp, BField, EField, physOPT, t0, tf, p_old, p_new, v_old, v_new)
		
		temp = qv_mult(qMult(qMult(qMult(qMult(qMult(qMult(rodriguezQuat(k9q, COEF['a109']*h), rodriguezQuat(k8q, COEF['a108']*h)), 
								   rodriguezQuat(k7q, COEF['a107']*h)), rodriguezQuat(k6q, COEF['a106']*h)), rodriguezQuat(k5q, COEF['a105']*h)), 
								   rodriguezQuat(k4q, COEF['a104']*h)), rodriguezQuat(k1q, COEF['a101']*h)), y)
		
		k10q = findCrossTerm(x+COEF['c10']*h, temp, BField, EField, physOPT, t0, tf, p_old, p_new, v_old, v_new)
		
		yy1q = qv_mult(qMult(qMult(qMult(qMult(qMult(qMult(qMult(rodriguezQuat(k10q, COEF['a1110']*h), rodriguezQuat(k9q, COEF['a119']*h)), 
								   rodriguezQuat(k8q, COEF['a118']*h)), rodriguezQuat(k7q, COEF['a117']*h)), rodriguezQuat(k6q, COEF['a116']*h)), 
								   rodriguezQuat(k5q, COEF['a115']*h)), rodriguezQuat(k4q, COEF['a114']*h)), rodriguezQuat(k1q, COEF['a111']*h)), y)
		
		k2q = findCrossTerm(x+COEF['c11']*h, yy1q, BField, EField, physOPT, t0, tf, p_old, p_new, v_old, v_new);
		xph = x + h;
		
		yy1q = qv_mult(qMult(qMult(qMult(qMult(qMult(qMult(qMult(qMult(rodriguezQuat(k2q, COEF['a1211']*h), rodriguezQuat(k10q, COEF['a1210']*h)), 
								   rodriguezQuat(k9q, COEF['a129']*h)), rodriguezQuat(k8q, COEF['a128']*h)), rodriguezQuat(k7q, COEF['a127']*h)), 
								   rodriguezQuat(k6q, COEF['a126']*h)), rodriguezQuat(k5q, COEF['a125']*h)), rodriguezQuat(k4q, COEF['a124']*h)),
								   rodriguezQuat(k1q, COEF['a121']*h)), y)
		k3q = findCrossTerm(xph, yy1q, BField, EField, physOPT, t0, tf, p_old, p_new, v_old, v_new);
		nfcn += 11;

		k4q = qMult(qMult(qMult(qMult(qMult(qMult(qMult(rodriguezQuat(k3q, h*COEF['b12']), rodriguezQuat(k2q, h*COEF['b11'])), rodriguezQuat(k10q, h*COEF['b10'])), 
						  rodriguezQuat(k9q, h*COEF['b9'])), rodriguezQuat(k8q, h*COEF['b8'])), rodriguezQuat(k7q, h*COEF['b7'])), 
						  rodriguezQuat(k6q, h*COEF['b6'])), rodriguezQuat(k1q, h*COEF['b1']))
		k5q = qv_mult(k4q, y) #k4 is the rotation matrix
		# error estimation 
		err = 0.0;
		err2 = 0.0;
		sk = atoli + rtoli * max_d3(np.abs(y), np.abs(k5q));
		erri = qv_mult(k4q, y) - qv_mult(qMult(qMult(rodriguezQuat(k3q, COEF['bhh3']*h), rodriguezQuat(k9q, COEF['bhh2']*h)), rodriguezQuat(k1q, COEF['bhh1']*h)), y)
		erri = np.max(np.square(erri))
		sqr = erri / sk;
		err2 += sum(sqr*sqr);
		erri = qv_mult(qMult(qMult(qMult(qMult(qMult(qMult(qMult(rodriguezQuat(k1q, COEF['er1']*h), rodriguezQuat(k6q, COEF['er6']*h)), rodriguezQuat(k7q, COEF['er7']*h)),
										 rodriguezQuat(k8q, COEF['er8']*h)), rodriguezQuat(k9q, COEF['er9']*h)), rodriguezQuat(k10q, COEF['er10']*h)),
										 rodriguezQuat(k2q, COEF['er11']*h)), rodriguezQuat(k3q, COEF['er12']*h)), y)
		erri = y-erri
		erri = np.max(np.square(erri));
		sqr = erri / sk;
		err += sum(sqr*sqr);
		deno = err + 0.01 * err2;
		if deno <= 0.0:
			deno = 1.0;
		err = abs(h) * err * np.sqrt (1.0 / (deno*n));
		# computation of hnew 
		fac11 = pow (err, expo1);
		# Lund-stabilization 
		fac = fac11 / pow(facold,np.float64(intOPT['beta']));
		# we require fac1 <= hnew/h <= fac2 
		fac = max (facc2, min (facc1, fac/np.float64(intOPT['safe'])));
		hnew = h / fac;
		if err <= 1.0:
			# step accepted 
			facold = max (err, 1.0E-4);
			naccpt+=1;
			y = k5q[:]
			nfcn+=1;
			if last:
				particle['s'] = y[:]
				particle['n_spin_steps'] += nstep
				particle['last_spin_step_size'] = outh
				return particle
			# final preparation for dense output 
			xold = x;
			x = xph;
		else:
			# step rejected
			hnew = h / min (facc1, fac11/np.float64(intOPT['safe']));
			reject = 1;
			if naccpt >= 1:
				nrejct=nrejct + 1;
			last = 0;
		h = hnew;
	particle['s'] = y[:]
	particle['n_spin_steps'] += nstep
	particle['last_spin_step_size'] = outh
	return particle

@numba.jit
def RK45(particle, intOPT, physOPT, BField, EField):
	t0 = particle['t_old']
	tf = particle['t']
	p_old = particle['x_old']
	p_new = particle['x']
	v_old = particle['v_old']
	v_new = particle['v']
	y = particle['s'][:]
	h = particle['last_spin_step_size']
	outh = h
	t = t0
	out = False
	stop = False
	nstep = 0
	while(1):
		nstep+=1
		h = min(h, intOPT['max_step']) #whichever is smaller use that
		h = max(h, intOPT['min_step']) 
		outh = h
		endOfSimulDt = tf - t #how long until the end of the simulation
		if endOfSimulDt <= h:
			stop = True
			h = endOfSimulDt
		#with the step size that is desired known, try to compute the step
		k1 = h*Bloch(t, y, BField, EField, physOPT, t0, tf, p_old, p_new, v_old, v_new)
		yy1 = y + RK45Coefs.B21*k1
		k2 = h*Bloch(t+RK45Coefs.A2*h, yy1, BField, EField, physOPT, t0, tf, p_old, p_new, v_old, v_new)
		yy1 = y + RK45Coefs.B31*k1 + RK45Coefs.B32*k2
		k3 = h*Bloch(t+RK45Coefs.A3*h, yy1, BField, EField, physOPT, t0, tf, p_old, p_new, v_old, v_new)
		yy1 = y + RK45Coefs.B41*k1 + RK45Coefs.B42*k2 + RK45Coefs.B43*k3
		k4 = h*Bloch(t+RK45Coefs.A4*h, yy1, BField, EField, physOPT, t0, tf, p_old, p_new, v_old, v_new)
		yy1 = y + RK45Coefs.B51*k1 + RK45Coefs.B52*k2 + RK45Coefs.B53*k3 + RK45Coefs.B54*k4
		k5 = h*Bloch(t+RK45Coefs.A5*h, yy1, BField, EField, physOPT, t0, tf, p_old, p_new, v_old, v_new)
		yy1 = y + RK45Coefs.B61*k1 + RK45Coefs.B62*k2 + RK45Coefs.B63*k3 + RK45Coefs.B64*k4 + RK45Coefs.B65*k5
		k6 = h*Bloch(t+RK45Coefs.A6*h, yy1, BField, EField, physOPT, t0, tf, p_old, p_new, v_old, v_new)
		weightedStep = y + k1*RK45Coefs.CH1 + k2*RK45Coefs.CH2 + k3*RK45Coefs.CH3+k4*RK45Coefs.CH4 + k5*RK45Coefs.CH5 + k6*RK45Coefs.CH6
		TE2 = np.abs(RK45Coefs.CT1*k1 + RK45Coefs.CT2*k2 + RK45Coefs.CT3*k3 + RK45Coefs.CT4*k4 + RK45Coefs.CT5*k5 + RK45Coefs.CT6*k6)
		absErr = np.max(np.abs(TE2))
		#print(t, h, y, weightedStep)
		if absErr  <= np.float64(intOPT['rtol']): #accept the step and move on to the next one
			t = t + np.float64(h) #what time are we at now
			y = weightedStep[:] #update the spin for the next iteration
		#now update the time step for the next calculation
		if absErr < 1.0E-16:
			absErr = 1.0E-16
		hnew = 0.9 * h * (intOPT['rtol']/absErr)**(1/5)
		h = hnew
		if stop:
			particle['s'] = y[:]
			particle['n_spin_steps'] += nstep
			particle['last_spin_step_size'] = outh
			return particle
	particle['s'] = y[:]
	particle['n_spin_steps'] += nstep
	particle['last_spin_step_size'] = outh
	return particle

@numba.jit
def RK45Rotation(particle, intOPT, physOPT, BField, EField):
	t0 = particle['t_old']
	tf = particle['t']
	p_old = particle['x_old']
	p_new = particle['x']
	v_old = particle['v_old']
	v_new = particle['v']
	y = particle['s'][:]
	h = particle['last_spin_step_size']
	t = t0
	stop = False
	hmin = 1.0E-9
	nstep = 0
	outh = particle['last_spin_step_size']
	while(1):
		nstep+=1
		endOfSimulDt = tf - t #how long until the end of the simulation
		outh = h
		if endOfSimulDt <= h:
			stop = True
			h = endOfSimulDt
		#with the step size that is desired known, try to compute the step
		k1 = findCrossTerm(t, y, BField, EField, physOPT, t0, tf, p_old, p_new, v_old, v_new)
		k2 = findCrossTerm(t+RK45Coefs.A2*h, appCross(y, k1, RK45Coefs.B21*h), BField, EField, physOPT, t0, tf, p_old, p_new, v_old, v_new)
		k3 = findCrossTerm(t+RK45Coefs.A3*h, appCross(appCross(y, k1, RK45Coefs.B31*h), k2, RK45Coefs.B32*h), BField, EField, physOPT, t0, tf, p_old, p_new, v_old, v_new)
		k4 = findCrossTerm(t+RK45Coefs.A4*h, appCross(appCross(appCross(y, k1, RK45Coefs.B41*h), k2, RK45Coefs.B42*h), k3, RK45Coefs.B43*h), BField, EField, physOPT, t0, tf, p_old, p_new, v_old, v_new)
		k5 = findCrossTerm(t+RK45Coefs.A5*h, appCross(appCross(appCross(appCross(y, k1, RK45Coefs.B51*h), k2, RK45Coefs.B52*h), 
																	   k3, RK45Coefs.B53*h), k4, RK45Coefs.B54*h), BField, EField, physOPT, t0, tf, p_old, p_new, v_old, v_new)
		k6 = findCrossTerm(t+RK45Coefs.A6*h, appCross(appCross(appCross(appCross(appCross(y, k1, RK45Coefs.B61*h), k2, RK45Coefs.B62*h), 
																	   k3, RK45Coefs.B63*h), k4, RK45Coefs.B64*h), k5, RK45Coefs.B65*h), BField, EField, physOPT, t0, tf, p_old, p_new, v_old, v_new)
		weightedStep = appCross(appCross(appCross(appCross(appCross(appCross(
			y, k1, h*RK45Coefs.CH1), k2, h*RK45Coefs.CH2), k3, h*RK45Coefs.CH3),
				k4, h*RK45Coefs.CH4), k5, h*RK45Coefs.CH5), k6, h*RK45Coefs.CH6)
		
		TE1 = appCross(appCross(appCross(appCross(appCross(
			y, k1, h*RK45Coefs.C1), k2, h*RK45Coefs.C2), k3, h*RK45Coefs.C3),
				k4, h*RK45Coefs.C4), k5, h*RK45Coefs.C5)
		
		TE2 = weightedStep - TE1
		#TE2 = np.identity(3) - rodriguez(k6, RK45Coefs.CT6*h)@(rodriguez(k5, RK45Coefs.CT5*h)@(rodriguez(k4, RK45Coefs.CT4*h)@(\
		#						rodriguez(k3, RK45Coefs.CT3*h)@(rodriguez(k2, RK45Coefs.CT2*h)@rodriguez(k1, RK45Coefs.CT1*h)))))
		#TE2 is meant to be equivalent to the null operation, basically the closer this is to the identity, the smaller the error
		#print(h, TE2)
		absErr = np.max(np.abs(TE2))
		if absErr  <= np.float64(intOPT['rtol']): #accept the step and move on to the next one
			t = t + np.float64(h) #what time are we at now
			y = weightedStep[:] #update the spin for the next iteration
		if absErr < 1.0E-16:
			absErr = 1.0E-16
		hnew = 0.9 * h * (intOPT['rtol']/absErr)**(1/5)
		h = hnew
		if stop:
			particle['s'] = y[:]
			particle['n_spin_steps'] += nstep
			particle['last_spin_step_size'] = outh
			return particle
	particle['s'] = y[:]
	particle['n_spin_steps'] += nstep
	particle['last_spin_step_size'] = outh
	return particle

@numba.jit
def RK45Quaternion(particle, intOPT, physOPT, BField, EField):
	t0 = particle['t_old']
	tf = particle['t']
	p_old = particle['x_old']
	p_new = particle['x']
	v_old = particle['v_old']
	v_new = particle['v']
	y = particle['s'][:]
	h = particle['last_spin_step_size']
	t = t0
	stop = False
	hmin = 1.0E-9
	nstep = 0
	outh = h
	while(1):
		nstep+=1
		endOfSimulDt = tf - t #how long until the end of the simulation
		h = min(h, intOPT['max_step']) #whichever is smaller use that
		h = max(h, intOPT['min_step']) 
		outh = h
		if endOfSimulDt <= h and endOfSimulDt <= h:
			stop = True
			h = endOfSimulDt
		#with the step size that is desired known, try to compute the step
		k1q = findCrossTerm(t, y, BField, EField, physOPT, t0, tf, p_old, p_new, v_old, v_new)
		
		temp = qv_mult(rodriguezQuat(k1q, RK45Coefs.B21*h), y)
		k2q = findCrossTerm(t+RK45Coefs.A2*h, temp, BField, EField, physOPT, t0, tf, p_old, p_new, v_old, v_new)
		
		temp = qv_mult(qMult(rodriguezQuat(k2q, RK45Coefs.B32*h), rodriguezQuat(k1q, RK45Coefs.B31*h)), y)
		k3q = findCrossTerm(t+RK45Coefs.A3*h, temp, BField, EField, physOPT, t0, tf, p_old, p_new, v_old, v_new)
		
		temp = qv_mult(qMult(rodriguezQuat(k3q, RK45Coefs.B43*h), qMult(rodriguezQuat(k2q, RK45Coefs.B42*h), rodriguezQuat(k1q, RK45Coefs.B41*h))), y)
		k4q = findCrossTerm(t+RK45Coefs.A4*h, temp, BField, EField, physOPT, t0, tf, p_old, p_new, v_old, v_new)
		
		temp = qv_mult(qMult(rodriguezQuat(k4q, RK45Coefs.B54*h), qMult(rodriguezQuat(k3q, RK45Coefs.B53*h), 
				qMult(rodriguezQuat(k2q, RK45Coefs.B52*h), rodriguezQuat(k1q, RK45Coefs.B51*h)))), y)
		k5q = findCrossTerm(t+RK45Coefs.A5*h, temp, BField, EField, physOPT, t0, tf, p_old, p_new, v_old, v_new)
		
		temp = qv_mult(qMult(rodriguezQuat(k5q, RK45Coefs.B65*h), qMult(rodriguezQuat(k4q, RK45Coefs.B64*h), 
				qMult(rodriguezQuat(k3q, RK45Coefs.B63*h), qMult(rodriguezQuat(k2q, RK45Coefs.B62*h), rodriguezQuat(k1q, RK45Coefs.B61*h))))), y)
		k6q = findCrossTerm(t+RK45Coefs.A6*h, temp, BField, EField, physOPT, t0, tf, p_old, p_new, v_old, v_new)
		weightedStepq = qv_mult(qMult(rodriguezQuat(k6q, h*RK45Coefs.CH6), qMult(rodriguezQuat(k5q, h*RK45Coefs.CH5), 
			qMult(rodriguezQuat(k4q, h*RK45Coefs.CH4), qMult(rodriguezQuat(k3q, h*RK45Coefs.CH3),
			qMult(rodriguezQuat(k2q, h*RK45Coefs.CH2), rodriguezQuat(k1q, h*RK45Coefs.CH1)))))), y)
		
		TE1 = qv_mult(qMult(rodriguezQuat(k5q, h*RK45Coefs.C5), qMult(rodriguezQuat(k4q, h*RK45Coefs.C4), 
							qMult(rodriguezQuat(k3q, h*RK45Coefs.C3), qMult(rodriguezQuat(k2q, h*RK45Coefs.C2), 
								rodriguezQuat(k1q, h*RK45Coefs.C1))))), y)
		TE2 = weightedStepq - TE1
		absErr = np.max(np.abs(TE2))
		if absErr  <= np.float64(intOPT['rtol']): #accept the step and move on to the next one
			t = t + np.float64(h) #what time are we at now
			y = weightedStepq[:] #update the spin for the next iteration
		if absErr < 1.0E-16:
			absErr = 1.0E-16
		hnew = 0.9 * h * (intOPT['rtol']/absErr)**(1/5)
		h = hnew
		if stop:
			particle['s'] = y[:]
			particle['n_spin_steps'] += nstep
			particle['last_spin_step_size'] = outh
			return particle
	particle['s'] = y[:]
	particle['n_spin_steps'] += nstep
	particle['last_spin_step_size'] = outh
	return particle

@numba.jit
def RK75109(particle, intOPT, physOPT, BField, EField):
	t0 = particle['t_old']
	tf = particle['t']
	p_old = particle['x_old']
	p_new = particle['x']
	v_old = particle['v_old']
	v_new = particle['v']
	y = particle['s'][:]
	h = particle['last_spin_step_size']
	coefs = __RK75109Coefs
	t = t0
	stop = False
	hmin = 1.0E-9
	nstep = 0
	outh = h
	while(1):
		nstep+=1
		endOfSimulDt = tf - t #how long until the end of the simulation
		outh = h
		if endOfSimulDt <= h:
			stop = True
			h = endOfSimulDt
		#with the step size that is desired known, try to compute the step
		k1 = h*Bloch(t, y, BField, EField, physOPT, t0, tf, p_old, p_new, v_old, v_new)
		yy1 = y + coefs['B21']*k1
		k2 = h*Bloch(t+coefs['A2']*h, yy1, BField, EField, physOPT, t0, tf, p_old, p_new, v_old, v_new)
		yy1 = y + coefs['B31']*k1 + coefs['B32']*k2
		k3 = h*Bloch(t+coefs['A3']*h, yy1, BField, EField, physOPT, t0, tf, p_old, p_new, v_old, v_new)
		yy1 = y + coefs['B41']*k1 + coefs['B42']*k2 + coefs['B43']*k3
		k4 = h*Bloch(t+coefs['A4']*h, yy1, BField, EField, physOPT, t0, tf, p_old, p_new, v_old, v_new)
		yy1 = y + coefs['B51']*k1 + coefs['B52']*k2 + coefs['B53']*k3 + coefs['B54']*k4
		k5 = h*Bloch(t+coefs['A5']*h, yy1, BField, EField, physOPT, t0, tf, p_old, p_new, v_old, v_new)
		yy1 = y + coefs['B61']*k1 + coefs['B62']*k2 + coefs['B63']*k3 + coefs['B64']*k4 + coefs['B65']*k5
		k6 = h*Bloch(t+coefs['A6']*h, yy1, BField, EField, physOPT, t0, tf, p_old, p_new, v_old, v_new)
		yy1 = y + coefs['B71']*k1 + coefs['B72']*k2 + coefs['B73']*k3 + coefs['B74']*k4 + coefs['B75']*k5 + coefs['B76']*k6
		k7 = h*Bloch(t+coefs['A7']*h, yy1, BField, EField, physOPT, t0, tf, p_old, p_new, v_old, v_new)
		yy1 = y + coefs['B81']*k1 + coefs['B82']*k2 + coefs['B83']*k3 + coefs['B84']*k4 + coefs['B85']*k5 + coefs['B86']*k6 + coefs['B87']*k7
		k8 = h*Bloch(t+coefs['A8']*h, yy1, BField, EField, physOPT, t0, tf, p_old, p_new, v_old, v_new)
		yy1 = y + coefs['B91']*k1 + coefs['B92']*k2 + coefs['B93']*k3 + coefs['B94']*k4 + coefs['B95']*k5 + coefs['B96']*k6 + coefs['B97']*k7 + coefs['B98']*k8
		k9 = h*Bloch(t+coefs['A9']*h, yy1, BField, EField, physOPT, t0, tf, p_old, p_new, v_old, v_new)
		
		weightedStep = y + k1*coefs['C1'] + k2*coefs['C2'] + k3*coefs['C3']+k4*coefs['C4'] + k5*coefs['C5'] + k6*coefs['C6'] + k7*coefs['C7'] + k8*coefs['C8'] + k9*coefs['C9']
		weightedStep2 = y + k1*coefs['CH1'] + k2*coefs['CH2'] + k3*coefs['CH3']+k4*coefs['CH4'] + k5*coefs['CH5'] + k6*coefs['CH6'] + k7*coefs['CH7'] + k8*coefs['CH8'] + k9*coefs['CH9']
		TE2 = weightedStep - weightedStep2
		absErr = np.max(np.abs(TE2))
		#print(t, h, y, weightedStep)
		if absErr  <= np.float64(intOPT['rtol']): #accept the step and move on to the next one
			t = t + np.float64(h) #what time are we at now
			y = weightedStep[:] #update the spin for the next iteration
		#now update the time step for the next calculation
		
		if absErr < 1.0E-16:
			absErr = 1.0E-16
		hnew = 0.9 * h * (intOPT['rtol']/absErr)**(1/5)
		h = hnew
		if stop:
			particle['s'] = y[:]
			particle['n_spin_steps'] += nstep
			return particle
	particle['s'] = y[:]
	particle['n_spin_steps'] += nstep
	particle['last_spin_step_size'] = outh
	return particle

@numba.jit
def RK75109Rotation(particle, intOPT, physOPT, BField, EField):
	t0 = particle['t_old']
	tf = particle['t']
	p_old = particle['x_old']
	p_new = particle['x']
	v_old = particle['v_old']
	v_new = particle['v']
	y = particle['s'][:]
	h = particle['last_spin_step_size']
	coefs = __RK75109Coefs
	t = t0
	stop = False
	hmin = 1.0E-9
	nstep = 0
	outh = h
	while(1):
		nstep+=1
		endOfSimulDt = tf - t #how long until the end of the simulation
		outh = h
		if endOfSimulDt <= h:
			stop = True
			h = endOfSimulDt
		#with the step size that is desired known, try to compute the step
		k1 = findCrossTerm(t, y, BField, EField, physOPT, t0, tf, p_old, p_new, v_old, v_new)
		yy1 = appCross(y, k1, coefs['B21']*h)
		
		k2 = findCrossTerm(t+coefs['A2']*h, yy1, BField, EField, physOPT, t0, tf, p_old, p_new, v_old, v_new)
		yy1 = appCross(appCross(y, k1, coefs['B31']*h), k2, coefs['B32']*h)
		
		k3 = findCrossTerm(t+coefs['A3']*h, yy1, BField, EField, physOPT, t0, tf, p_old, p_new, v_old, v_new)
		yy1 = appCross(appCross(appCross(y, k1, coefs['B41']*h), k2, coefs['B42']*h), k3, coefs['B43']*h)
		
		k4 = findCrossTerm(t+coefs['A4']*h, yy1, BField, EField, physOPT, t0, tf, p_old, p_new, v_old, v_new)
		yy1 = appCross(appCross(appCross(appCross(y, k1, coefs['B51']*h), k2, coefs['B52']*h), k3, coefs['B53']*h), k4, coefs['B54']*h)
		
		k5 = findCrossTerm(t+coefs['A5']*h, yy1, BField, EField, physOPT, t0, tf, p_old, p_new, v_old, v_new)
		yy1 = appCross(appCross(appCross(appCross(appCross(y, k1, coefs['B61']*h), k2, coefs['B62']*h), k3, coefs['B63']*h), k4, coefs['B64']*h), k5, coefs['B65']*h)
		
		k6 = findCrossTerm(t+coefs['A6']*h, yy1, BField, EField, physOPT, t0, tf, p_old, p_new, v_old, v_new)
		yy1 = appCross(appCross(appCross(appCross(appCross(appCross(
			y, k1, coefs['B71']*h), k2, coefs['B72']*h), k3, coefs['B73']*h), k4, coefs['B74']*h), k5, coefs['B75']*h), k6, coefs['B76']*h)
		
		k7 = findCrossTerm(t+coefs['A7']*h, yy1, BField, EField, physOPT, t0, tf, p_old, p_new, v_old, v_new)
		yy1 = appCross(appCross(appCross(appCross(appCross(appCross(appCross(
			y, k1, coefs['B81']*h), k2, coefs['B82']*h), k3, coefs['B83']*h), k4, coefs['B84']*h), k5, coefs['B85']*h), k6, coefs['B86']*h), k7, coefs['B87']*h)
		
		k8 = findCrossTerm(t+coefs['A8']*h, yy1, BField, EField, physOPT, t0, tf, p_old, p_new, v_old, v_new)
		yy1 = appCross(appCross(appCross(appCross(appCross(appCross(appCross(appCross(
			y, k1, coefs['B91']*h), k2, coefs['B92']*h), k3, coefs['B93']*h), k4, coefs['B94']*h), k5, coefs['B95']*h), k6, coefs['B96']*h), k7, coefs['B97']*h), k8, coefs['B98']*h)
		
		k9 = findCrossTerm(t+coefs['A9']*h, yy1, BField, EField, physOPT, t0, tf, p_old, p_new, v_old, v_new)
		
		weightedStep = appCross(appCross(appCross(appCross(appCross(appCross(appCross(appCross(appCross(
			y, k1, coefs['C1']*h), k2, coefs['C2']*h), k3, coefs['C3']*h), k4, coefs['C4']*h), k5, coefs['C5']*h), k6, coefs['C6']*h), k7, coefs['C7']*h), k8, coefs['C8']*h), k9, coefs['C9']*h)
		
		weightedStep2 = appCross(appCross(appCross(appCross(appCross(appCross(appCross(appCross(appCross(
			y, k1, coefs['CH1']*h), k2, coefs['CH2']*h), k3, coefs['CH3']*h), k4, coefs['CH4']*h), k5, coefs['CH5']*h), k6, coefs['CH6']*h), k7, coefs['CH7']*h), k8, coefs['CH8']*h), k9, coefs['CH9']*h)
		
		TE2 = weightedStep - weightedStep2
		absErr = np.max(np.abs(TE2))
		#print(t, h, y, weightedStep)
		if absErr  <= np.float64(intOPT['rtol']): #accept the step and move on to the next one
			t = t + np.float64(h) #what time are we at now
			y = weightedStep[:] #update the spin for the next iteration
		if absErr < 1.0E-16:
			absErr = 1.0E-16
		hnew = 0.9 * h * (intOPT['rtol']/absErr)**(1/5)
		h = hnew
		if stop:
			particle['s'] = y[:]
			particle['n_spin_steps'] += nstep
			particle['last_spin_step_size'] = outh
			return particle
	particle['s'] = y[:]
	particle['n_spin_steps'] += nstep
	particle['last_spin_step_size'] = outh
	return particle

@numba.jit
def ImplicitEuler(particle, intOPT, physOPT, BField, EField):
	t0 = particle['t_old']
	tf = particle['t']
	p_old = particle['x_old']
	p_new = particle['x']
	v_old = particle['v_old']
	v_new = particle['v']
	y = particle['s'][:]
	h = np.float64(intOPT['h']) #for this integrator, just always use the fixed step size
	t = t0
	stop = False
	nstep = 0
	while not stop:
		nstep+=1
		endOfSimulDt = tf - t #how long until the end of the simulation
		if endOfSimulDt <= h:
			stop = True
			h = endOfSimulDt
		nextTime = t+h
		cross = findCrossTerm(nextTime, y, BField, EField, physOPT, t0, tf, p_old, p_new, v_old, v_new)
		matrix = determineMatrix(cross)
		denominator = np.identity(3) - h*matrix
		inverse = np.linalg.inv(denominator)
		y = vecNorm(inverse@y)
		t = nextTime
	particle['s'] = y[:]
	particle['n_spin_steps'] += nstep
	return particle

@numba.jit
def CrankNicolson(particle, intOPT, physOPT, BField, EField):
	t0 = particle['t_old']
	tf = particle['t']
	p_old = particle['x_old']
	p_new = particle['x']
	v_old = particle['v_old']
	v_new = particle['v']
	y = particle['s'][:]
	h = np.float64(intOPT['h']) #for this integrator, just always use the fixed step size
	t = t0
	stop = False
	nstep = 0
	while not stop:
		nstep+=1
		endOfSimulDt = tf - t #how long until the end of the simulation
		if endOfSimulDt <= h:
			stop = True
			h = endOfSimulDt
		nextTime = t+h
		crossNow = findCrossTerm(t, y, BField, EField, physOPT, t0, tf, p_old, p_new, v_old, v_new)
		crossNext = findCrossTerm(nextTime, y, BField, EField, physOPT, t0, tf, p_old, p_new, v_old, v_new)
		matrixNow = determineMatrix(crossNow)
		matrixNext = determineMatrix(crossNext)
		numerator = np.identity(3)+0.5*h*matrixNow
		denominator = np.identity(3) - 0.5*h*matrixNext
		y = np.linalg.inv(denominator)@numerator@y
		t = nextTime
	particle['s'] = y[:]
	particle['n_spin_steps'] += nstep
	return particle

@numba.jit
def perturbativeIntegratorRK45(particle, intOPT, physOPT, BField, EField):
	t0 = particle['t_old']
	tf = particle['t']
	p_old = particle['x_old']
	p_new = particle['x']
	v_old = particle['v_old']
	v_new = particle['v']
	y = particle['s'][:]
	h = particle['last_spin_step_size']
	t = t0
	stop = False
	hmin = 1.0E-9
	nstep = 0
	outh = particle['last_spin_step_size']
	stop = False
	while not stop:
		nstep+=1
		#print(t, h
		endOfSimulDt = tf - t #how long until the end of the simulation
		outh = h
		if endOfSimulDt <= h:
			stop = True
			h = endOfSimulDt
		#with the step size that is desired known, try to compute the step
		#first calculate where the particle will end up based on assuming no particle parameters change
		initialCrossTerm = findCrossTerm(t, y, BField, EField, physOPT, t0, tf, p_old, p_new, v_old, v_new)
		finalStateInitGuess = appCross(y, initialCrossTerm, h)
		
		perturbation = findCrossTerm(t+h, y, BField, EField, physOPT, t0, tf, p_old, p_new, v_old, v_new)
		deviation = initialCrossTerm - perturbation
		matrix = rodriguez(deviation, h)
		if np.any(np.isnan(matrix)):
			y = finalStateInitGuess[:]
			t = t + h
			h = h * 1.1
		else:
			corrected = matrix@finalStateInitGuess
			error = np.max(np.abs(corrected - finalStateInitGuess))
			print(t, h, matrix)
			if error < intOPT['rtol']:
				#step is accepted
				y = corrected[:]
				t = t + h
				h = h * 1.1
			else:
				h = h/2.0
	particle['s'] = y[:]
	particle['n_spin_steps'] += nstep
	particle['last_spin_step_size'] = outh
	return particle

@numba.jit
def integrateSpin(particle, simulationOptions, spinOptions, BField, EField):
	if particle['last_spin_step_size'] <= 1.0e-9: #if the last step size was less than 1 nanosecond, basically it probably hadn't gone yet
		particle['last_spin_step_size'] = spinOptions['h']
	if spinOptions['integrator'] == 0:#traditional DOP853 method
		particle = DOP853(particle, spinOptions, simulationOptions, BField, EField)
	elif spinOptions['integrator'] == 1:
		particle = DOP853Rotation(particle, spinOptions, simulationOptions, BField, EField)
	elif spinOptions['integrator'] == 2:
		particle = DOP853Quaternion(particle, spinOptions, simulationOptions, BField, EField)
	elif spinOptions['integrator'] == 3:
		particle = RK45(particle, spinOptions, simulationOptions, BField, EField)
	elif spinOptions['integrator'] == 4:
		particle = RK45Rotation(particle, spinOptions, simulationOptions, BField, EField)
	elif spinOptions['integrator'] == 5:
		particle = RK45Quaternion(particle, spinOptions, simulationOptions, BField, EField)
	elif spinOptions['integrator'] == 6:
		particle = RK75109(particle, spinOptions, simulationOptions, BField, EField)
	elif spinOptions['integrator'] == 7:
		particle = RK75109Rotation(particle, spinOptions, simulationOptions, BField, EField)
	elif spinOptions['integrator'] == 8:
		None
	elif spinOptions['integrator'] == 9:
		particle = ImplicitEuler(particle, spinOptions, simulationOptions, BField, EField)
	elif spinOptions['integrator'] == 10:
		particle = CrankNicolson(particle, spinOptions, simulationOptions, BField, EField)
	elif spinOptions['integrator'] == 11:
		particle = perturbativeIntegratorRK45(particle, spinOptions, simulationOptions, BField, EField)
	return particle