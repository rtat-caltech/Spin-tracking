'''
This file is a bunch of Python-based spin integration routines. 

All of these should be defined so they are interchangeable if possible.
'''

import numpy as np
import numba

@numba.jit
def __DOP853Coefs():
	COEF = {
		"c2" : 0.526001519587677318785587544488E-01,
		"c3" : 0.789002279381515978178381316732E-01,
		"c4" : 0.118350341907227396726757197510E+00,
		"c5" : 0.281649658092772603273242802490E+00,
		"c6" : 0.333333333333333333333333333333E+00,
		"c7" : 0.25E+00,
		"c8" : 0.307692307692307692307692307692E+00,
		"c9" : 0.651282051282051282051282051282E+00,
		"c10" : 0.6E+00,
		"c11" : 0.857142857142857142857142857142E+00,
		"c14" : 0.1E+00,
		"c15" : 0.2E+00,
		"c16" : 0.777777777777777777777777777778E+00,

		"b1" :   5.42937341165687622380535766363E-2,
		"b6" :   4.45031289275240888144113950566E0,
		"b7" :   1.89151789931450038304281599044E0,
		"b8" :  -5.8012039600105847814672114227E0,
		"b9" :   3.1116436695781989440891606237E-1,
		"b10" : -1.52160949662516078556178806805E-1,
		"b11" :  2.01365400804030348374776537501E-1,
		"b12" :  4.47106157277725905176885569043E-2,

		"bhh1" : 0.244094488188976377952755905512E+00,
		"bhh2" : 0.733846688281611857341361741547E+00,
		"bhh3" : 0.220588235294117647058823529412E-01,

		"er1" :  0.1312004499419488073250102996E-01,
		"er6" : -0.1225156446376204440720569753E+01,
		"er7" : -0.4957589496572501915214079952E+00,
		"er8" :  0.1664377182454986536961530415E+01,
		"er9" : -0.3503288487499736816886487290E+00,
		"er10" :  0.3341791187130174790297318841E+00,
		"er11" :  0.8192320648511571246570742613E-01,
		"er12" : -0.2235530786388629525884427845E-01,

		"a21" :    5.26001519587677318785587544488E-2,
		"a31" :    1.97250569845378994544595329183E-2,
		"a32" :    5.91751709536136983633785987549E-2,
		"a41" :    2.95875854768068491816892993775E-2,
		"a43" :    8.87627564304205475450678981324E-2,
		"a51" :    2.41365134159266685502369798665E-1,
		"a53" :   -8.84549479328286085344864962717E-1,
		"a54" :    9.24834003261792003115737966543E-1,
		"a61" :    3.7037037037037037037037037037E-2,
		"a64" :    1.70828608729473871279604482173E-1,
		"a65" :    1.25467687566822425016691814123E-1,
		"a71" :    3.7109375E-2,
		"a74" :    1.70252211019544039314978060272E-1,
		"a75" :    6.02165389804559606850219397283E-2,
		"a76" :   -1.7578125E-2,

		"a81" :    3.70920001185047927108779319836E-2,
		"a84" :    1.70383925712239993810214054705E-1,
		"a85" :    1.07262030446373284651809199168E-1,
		"a86" :   -1.53194377486244017527936158236E-2,
		"a87" :    8.27378916381402288758473766002E-3,
		"a91" :    6.24110958716075717114429577812E-1,
		"a94" :   -3.36089262944694129406857109825E0,
		"a95" :   -8.68219346841726006818189891453E-1,
		"a96" :    2.75920996994467083049415600797E1,
		"a97" :    2.01540675504778934086186788979E1,
		"a98" :   -4.34898841810699588477366255144E1,
		"a101" :   4.77662536438264365890433908527E-1,
		"a104" :  -2.48811461997166764192642586468E0,
		"a105" :  -5.90290826836842996371446475743E-1,
		"a106" :   2.12300514481811942347288949897E1,
		"a107" :   1.52792336328824235832596922938E1,
		"a108" :  -3.32882109689848629194453265587E1,
		"a109" :  -2.03312017085086261358222928593E-2,

		"a111" :  -9.3714243008598732571704021658E-1,
		"a114" :   5.18637242884406370830023853209E0,
		"a115" :   1.09143734899672957818500254654E0,
		"a116" :  -8.14978701074692612513997267357E0,
		"a117" :  -1.85200656599969598641566180701E1,
		"a118" :   2.27394870993505042818970056734E1,
		"a119" :   2.49360555267965238987089396762E0,
		"a1110" : -3.0467644718982195003823669022E0,
		"a121" :   2.27331014751653820792359768449E0,
		"a124" :  -1.05344954667372501984066689879E1,
		"a125" :  -2.00087205822486249909675718444E0,
		"a126" :  -1.79589318631187989172765950534E1,
		"a127" :   2.79488845294199600508499808837E1,
		"a128" :  -2.85899827713502369474065508674E0,
		"a129" :  -8.87285693353062954433549289258E0,
		"a1210" :  1.23605671757943030647266201528E1,
		"a1211" :  6.43392746015763530355970484046E-1,

		"a141" :  5.61675022830479523392909219681E-2,
		"a147" :  2.53500210216624811088794765333E-1,
		"a148" : -2.46239037470802489917441475441E-1,
		"a149": -1.24191423263816360469010140626E-1,
		"a1410" :  1.5329179827876569731206322685E-1,
		"a1411" :  8.20105229563468988491666602057E-3,
		"a1412" :  7.56789766054569976138603589584E-3,
		"a1413" : -8.298E-3,

		"a151" :  3.18346481635021405060768473261E-2,
		"a156" :  2.83009096723667755288322961402E-2,
		"a157" :  5.35419883074385676223797384372E-2,
		"a158" : -5.49237485713909884646569340306E-2,
		"a1511" : -1.08347328697249322858509316994E-4,
		"a1512" :  3.82571090835658412954920192323E-4,
		"a1513" : -3.40465008687404560802977114492E-4,
		"a1514" :  1.41312443674632500278074618366E-1,
		"a161" : -4.28896301583791923408573538692E-1,
		"a166" : -4.69762141536116384314449447206E0,
		"a167" :  7.68342119606259904184240953878E0,
		"a168" :  4.06898981839711007970213554331E0,
		"a169" :  3.56727187455281109270669543021E-1,
		"a1613" : -1.39902416515901462129418009734E-3,
		"a1614" :  2.9475147891527723389556272149E0,
		"a1615" : -9.15095847217987001081870187138E0,

		"d41" : -0.84289382761090128651353491142E+01,
		"d46" :  0.56671495351937776962531783590E+00,
		"d47" : -0.30689499459498916912797304727E+01,
		"d48" :  0.23846676565120698287728149680E+01,
		"d49" :  0.21170345824450282767155149946E+01,
		"d410" : -0.87139158377797299206789907490E+00,
		"d411" :  0.22404374302607882758541771650E+01,
		"d412" :  0.63157877876946881815570249290E+00,
		"d413" : -0.88990336451333310820698117400E-01,
		"d414" :  0.18148505520854727256656404962E+02,
		"d415" : -0.91946323924783554000451984436E+01,
		"d416" : -0.44360363875948939664310572000E+01,

		"d51" :  0.10427508642579134603413151009E+02,
		"d56" :  0.24228349177525818288430175319E+03,
		"d57" :  0.16520045171727028198505394887E+03,
		"d58" : -0.37454675472269020279518312152E+03,
		"d59" : -0.22113666853125306036270938578E+02,
		"d510" :  0.77334326684722638389603898808E+01,
		"d511" : -0.30674084731089398182061213626E+02,
		"d512" : -0.93321305264302278729567221706E+01,
		"d513" :  0.15697238121770843886131091075E+02,
		"d514" : -0.31139403219565177677282850411E+02,
		"d515" : -0.93529243588444783865713862664E+01,
		"d516" :  0.35816841486394083752465898540E+02,

		"d61" :  0.19985053242002433820987653617E+02,
		"d66" : -0.38703730874935176555105901742E+03,
		"d67" : -0.18917813819516756882830838328E+03,
		"d68" :  0.52780815920542364900561016686E+03,
		"d69" : -0.11573902539959630126141871134E+02,
		"d610" :  0.68812326946963000169666922661E+01,
		"d611" : -0.10006050966910838403183860980E+01,
		"d612" :  0.77771377980534432092869265740E+00,
		"d613" : -0.27782057523535084065932004339E+01,
		"d614" : -0.60196695231264120758267380846E+02,
		"d615" :  0.84320405506677161018159903784E+02,
		"d616" :  0.11992291136182789328035130030E+02,

		"d71" : -0.25693933462703749003312586129E+02,
		"d76" : -0.15418974869023643374053993627E+03,
		"d77" : -0.23152937917604549567536039109E+03,
		"d78":  0.35763911791061412378285349910E+03,
		"d79":  0.93405324183624310003907691704E+02,
		"d710" : -0.37458323136451633156875139351E+02,
		"d711" :  0.10409964950896230045147246184E+03,
		"d712" :  0.29840293426660503123344363579E+02,
		"d713" : -0.43533456590011143754432175058E+02,
		"d714" :  0.96324553959188282948394950600E+02,
		"d715" : -0.39177261675615439165231486172E+02,
		"d716" : -0.14972683625798562581422125276E+03
	}
	return COEF

@numba.jit
def __rkParamDef():
	rk45COEF = {
		'A1': 0.0,
		'A2': 2.0/9.0,
		'A3': 1.0/3.0,
		'A4': 3.0/4.0,
		'A5': 1.0,
		'A6': 5.0/6.0,
		'B21': 2.0/9.0,
		'B31': 1.0/12.0,
		'B32': 1.0/4.0,
		'B41': 69/128,
		'B42': -243/128,
		'B43': 135/64,
		'B51': -17/12,
		'B52': 27/4,
		'B53': -27/5,
		'B54': 16/15,
		'B61': 65/432,
		'B62': -5/16,
		'B63': 13/16,
		'B64': 4/27,
		'B65': 5/144,
		'C1': 1/9,
		'C2': 0,
		'C3': 9/20,
		'C4': 16/45,
		'C5': 1/12,
		'CH1': 47/450,
		'CH2': 0,
		'CH3': 12/25,
		'CH4': 32/225,
		'CH5': 1/30,
		'CH6': 6/25,
		'CT1': 1/150,
		'CT2': 0,
		'CT3': -3/100,
		'CT4': 16/75,
		'CT5': 1/20,
		'CT6':-6/25,
	}
	return rk45COEF

@numba.jit
def __RK75109Coefs():
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
	COEF = __DOP853Coefs()
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
	h = np.float64(intOPT['h'])
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
	while 1:
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
		nstep+=1
		yy1 = y + h * COEF['a21'] * k1;
		k2 = Bloch(x+COEF['c2']*h, yy1, BField, EField, physOPT, t0, tf, p_old, p_new, v_old, v_new);

		yy1 = y + h * (COEF['a31']*k1 + COEF['a32']*k2);
		k3 = Bloch(x+COEF['c3']*h, yy1, BField, EField, physOPT, t0, tf, p_old, p_new, v_old, v_new);

		yy1 = y + h * (COEF['a41']*k1 + COEF['a43']*k3);
		k4 = Bloch(x+COEF['c4']*h, yy1, BField, EField, physOPT, t0, tf, p_old, p_new, v_old, v_new);

		yy1 = y + h * (COEF['a51']*k1 + COEF['a53']*k3 + COEF['a54']*k4);
		k5 = Bloch(x+COEF['c5']*h, yy1, BField, EField, physOPT, t0, tf, p_old, p_new, v_old, v_new);

		yy1 = y + h * (COEF['a61']*k1 + COEF['a64']*k4 + COEF['a65']*k5);
		k6 = Bloch(x+COEF['c6']*h, yy1, BField, EField, physOPT, t0, tf, p_old, p_new, v_old, v_new);

		yy1 = y + h * (COEF['a71']*k1 + COEF['a74']*k4 + COEF['a75']*k5 + COEF['a76']*k6);
		k7 = Bloch(x+COEF['c7']*h, yy1, BField, EField, physOPT, t0, tf, p_old, p_new, v_old, v_new);

		yy1 = y + h * (COEF['a81']*k1 + COEF['a84']*k4 + COEF['a85']*k5 + COEF['a86']*k6 + COEF['a87']*k7);
		k8 = Bloch(x+COEF['c8']*h, yy1, BField, EField, physOPT, t0, tf, p_old, p_new, v_old, v_new);

		yy1 = y + h * (COEF['a91']*k1 + COEF['a94']*k4 + COEF['a95']*k5 + COEF['a96']*k6 + COEF['a97']*k7 + COEF['a98']*k8);
		k9 = Bloch(x+COEF['c9']*h, yy1, BField, EField, physOPT, t0, tf, p_old, p_new, v_old, v_new);

		yy1 = y + h * (COEF['a101']*k1 + COEF['a104']*k4 + COEF['a105']*k5 + COEF['a106']*k6 + COEF['a107']*k7 + COEF['a108']*k8 + COEF['a109']*k9);
		k10 = Bloch(x+COEF['c10']*h, yy1, BField, EField, physOPT, t0, tf, p_old, p_new, v_old, v_new);

		yy1 = y + h * (COEF['a111']*k1 + COEF['a114']*k4 + COEF['a115']*k5 + COEF['a116']*k6 + COEF['a117']*k7 + COEF['a118']*k8 + COEF['a119']*k9 + COEF['a1110']*k10);

		k2 = Bloch(x+COEF['c11']*h, yy1, BField, EField, physOPT, t0, tf, p_old, p_new, v_old, v_new);
		xph = x + h;

		yy1 = y + h * (COEF['a121']*k1 + COEF['a124']*k4 + COEF['a125']*k5 + COEF['a126']*k6 + COEF['a127']*k7 + COEF['a128']*k8 + COEF['a129']*k9 + COEF['a1210']*k10 + COEF['a1211']*k2);

		k3 = Bloch(xph, yy1, BField, EField, physOPT, t0, tf, p_old, p_new, v_old, v_new);
		nfcn += 11;

		k4 = COEF['b1']*k1 + COEF['b6']*k6 + COEF['b7']*k7 + COEF['b8']*k8 + COEF['b9']*k9 + COEF['b10']*k10 + COEF['b11']*k2 + COEF['b12']*k3;
		k5 = y + h * k4;

		# error estimation 
		err = 0.0;
		err2 = 0.0;
		sk = atoli + rtoli * max_d3(np.abs(y), np.abs(k5));
		erri = k4 - np.float64(COEF['bhh1'])*k1 - np.float64(COEF['bhh2'])*k9 - np.float64(COEF['bhh3'])*k3;
		sqr = erri / sk;
		err2 += sum(sqr*sqr);
		erri = np.float64(COEF['er1'])*k1 + np.float64(COEF['er6'])*k6 + np.float64(COEF['er7'])*k7 + np.float64(COEF['er8'])*k8 +\
			np.float64(COEF['er9'])*k9 + np.float64(COEF['er10'])*k10 + np.float64(COEF['er11'])*k2 + np.float64(COEF['er12'])*k3;
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
	COEF = __DOP853Coefs()
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
	h = np.float64(intOPT['h'])
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
	COEF = __DOP853Coefs()
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
	h = np.float64(intOPT['h'])
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
		#erri = k4 - COEF['bhh1']*k1 - COEF['bhh2']*k9 - COEF['bhh3']*k3
		#This one is a different method of finding the endpoint based on previously calculated values
		#erri = k5q - appCross(appCross(appCross(y, k1q, COEF['bhh1']*h), k9q, COEF['bhh2']*h), k3q, COEF['bhh3']*h)
		erri = qv_mult(k4q, y) - qv_mult(qMult(qMult(rodriguezQuat(k3q, COEF['bhh3']*h), rodriguezQuat(k9q, COEF['bhh2']*h)), rodriguezQuat(k1q, COEF['bhh1']*h)), y)
		#erri = k4@y - (rodriguez(k1, COEF['bhh1']*h)@(rodriguez(k9, COEF['bhh2']*h)@(rodriguez(k3, COEF['bhh3']*h))))@y;
		#print(erri)
		erri = np.max(np.square(erri))
		sqr = erri / sk;
		err2 += sum(sqr*sqr);
		#erri = COEF['er1']*k1 + COEF['er6']*k6 + COEF['er7']*k7 + COEF['er8']*k8 + COEF['er9']*k9 + COEF['er10']*k10 + COEF['er11']*k2 + COEF['er12']*k3;
		#erri = (np.identity(3) - rodriguez(k1q, COEF['er1']*h)@(rodriguez(k6q, COEF['er6']*h)@(rodriguez(k7q, COEF['er7']*h)@(\
		#	rodriguez(k8q, COEF['er8']*h)@(rodriguez(k9q, COEF['er9']*h)@(rodriguez(k10q, COEF['er10']*h)@(rodriguez(k2q, COEF['er11']*h)@(rodriguez(k3q, COEF['er12']*h)))))))))@y
		#erri = appCross(appCross(appCross(appCross(appCross(appCross(appCross(appCross(
		#	y, k3q, COEF['er12']*h), k2q, COEF['er11']*h), k10q, COEF['er10']*h), k9q, COEF['er9']*h), k8q, COEF['er8']*h), 
		#								  k7q, COEF['er7']*h), k6q, COEF['er6']*h), k1q, COEF['er1']*h)
		
		erri = qv_mult(qMult(qMult(qMult(qMult(qMult(qMult(qMult(rodriguezQuat(k1q, COEF['er1']*h), rodriguezQuat(k6q, COEF['er6']*h)), rodriguezQuat(k7q, COEF['er7']*h)),
										 rodriguezQuat(k8q, COEF['er8']*h)), rodriguezQuat(k9q, COEF['er9']*h)), rodriguezQuat(k10q, COEF['er10']*h)),
										 rodriguezQuat(k2q, COEF['er11']*h)), rodriguezQuat(k3q, COEF['er12']*h)), y)
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
			if last:
				particle['s'] = y[:]
				particle['n_spin_steps'] += nstep
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
	h = np.float64(intOPT['h'])
	rk45COEF = __rkParamDef()
	t = t0
	out = False
	stop = False
	hmin = 1.0E-9
	nstep = 0
	while(1):
		nstep+=1
		h = min(h, intOPT['max_step']) #whichever is smaller use that
		h = max(h, intOPT['min_step']) 
		endOfSimulDt = tf - t #how long until the end of the simulation
		if endOfSimulDt <= h:
			stop = True
			h = endOfSimulDt
		#with the step size that is desired known, try to compute the step
		k1 = h*Bloch(t, y, BField, EField, physOPT, t0, tf, p_old, p_new, v_old, v_new)
		yy1 = y + rk45COEF['B21']*k1
		k2 = h*Bloch(t+rk45COEF['A2']*h, yy1, BField, EField, physOPT, t0, tf, p_old, p_new, v_old, v_new)
		yy1 = y + rk45COEF['B31']*k1 + rk45COEF['B32']*k2
		k3 = h*Bloch(t+rk45COEF['A3']*h, yy1, BField, EField, physOPT, t0, tf, p_old, p_new, v_old, v_new)
		yy1 = y + rk45COEF['B41']*k1 + rk45COEF['B42']*k2 + rk45COEF['B43']*k3
		k4 = h*Bloch(t+rk45COEF['A4']*h, yy1, BField, EField, physOPT, t0, tf, p_old, p_new, v_old, v_new)
		yy1 = y + rk45COEF['B51']*k1 + rk45COEF['B52']*k2 + rk45COEF['B53']*k3 + rk45COEF['B54']*k4
		k5 = h*Bloch(t+rk45COEF['A5']*h, yy1, BField, EField, physOPT, t0, tf, p_old, p_new, v_old, v_new)
		yy1 = y + rk45COEF['B61']*k1 + rk45COEF['B62']*k2 + rk45COEF['B63']*k3 + rk45COEF['B64']*k4 + rk45COEF['B65']*k5
		k6 = h*Bloch(t+rk45COEF['A6']*h, yy1, BField, EField, physOPT, t0, tf, p_old, p_new, v_old, v_new)
		weightedStep = y + k1*rk45COEF['CH1'] + k2*rk45COEF['CH2'] + k3*rk45COEF['CH3']+k4*rk45COEF['CH4'] + k5*rk45COEF['CH5'] + k6*rk45COEF['CH6']
		TE2 = np.abs(rk45COEF['CT1']*k1 + rk45COEF['CT2']*k2 + rk45COEF['CT3']*k3 + rk45COEF['CT4']*k4 + rk45COEF['CT5']*k5 + rk45COEF['CT6']*k6)
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
	h = np.float64(intOPT['h'])
	rk45COEF = __rkParamDef()
	t = t0
	stop = False
	hmin = 1.0E-9
	nstep = 0
	while(1):
		nstep+=1
		endOfSimulDt = tf - t #how long until the end of the simulation
		if endOfSimulDt <= h:
			stop = True
			h = endOfSimulDt
		#with the step size that is desired known, try to compute the step
		k1 = findCrossTerm(t, y, BField, EField, physOPT, t0, tf, p_old, p_new, v_old, v_new)
		k2 = findCrossTerm(t+rk45COEF['A2']*h, appCross(y, k1, rk45COEF['B21']*h), BField, EField, physOPT, t0, tf, p_old, p_new, v_old, v_new)
		k3 = findCrossTerm(t+rk45COEF['A3']*h, appCross(appCross(y, k1, rk45COEF['B31']*h), k2, rk45COEF['B32']*h), BField, EField, physOPT, t0, tf, p_old, p_new, v_old, v_new)
		k4 = findCrossTerm(t+rk45COEF['A4']*h, appCross(appCross(appCross(y, k1, rk45COEF['B41']*h), k2, rk45COEF['B42']*h), k3, rk45COEF['B43']*h), BField, EField, physOPT, t0, tf, p_old, p_new, v_old, v_new)
		k5 = findCrossTerm(t+rk45COEF['A5']*h, appCross(appCross(appCross(appCross(y, k1, rk45COEF['B51']*h), k2, rk45COEF['B52']*h), 
																	   k3, rk45COEF['B53']*h), k4, rk45COEF['B54']*h), BField, EField, physOPT, t0, tf, p_old, p_new, v_old, v_new)
		k6 = findCrossTerm(t+rk45COEF['A6']*h, appCross(appCross(appCross(appCross(appCross(y, k1, rk45COEF['B61']*h), k2, rk45COEF['B62']*h), 
																	   k3, rk45COEF['B63']*h), k4, rk45COEF['B64']*h), k5, rk45COEF['B65']*h), BField, EField, physOPT, t0, tf, p_old, p_new, v_old, v_new)
		weightedStep = appCross(appCross(appCross(appCross(appCross(appCross(
			y, k1, h*rk45COEF['CH1']), k2, h*rk45COEF['CH2']), k3, h*rk45COEF['CH3']),
				k4, h*rk45COEF['CH4']), k5, h*rk45COEF['CH5']), k6, h*rk45COEF['CH6'])
		
		TE1 = appCross(appCross(appCross(appCross(appCross(
			y, k1, h*rk45COEF['C1']), k2, h*rk45COEF['C2']), k3, h*rk45COEF['C3']),
				k4, h*rk45COEF['C4']), k5, h*rk45COEF['C5'])
		
		TE2 = weightedStep - TE1
		#TE2 = np.identity(3) - rodriguez(k6, rk45COEF['CT6']*h)@(rodriguez(k5, rk45COEF['CT5']*h)@(rodriguez(k4, rk45COEF['CT4']*h)@(\
		#						rodriguez(k3, rk45COEF['CT3']*h)@(rodriguez(k2, rk45COEF['CT2']*h)@rodriguez(k1, rk45COEF['CT1']*h)))))
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
			return particle
	particle['s'] = y[:]
	particle['n_spin_steps'] += nstep
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
	h = np.float64(intOPT['h'])
	rk45COEF = __rkParamDef()
	t = t0
	stop = False
	hmin = 1.0E-9
	nstep = 0
	while(1):
		nstep+=1
		endOfSimulDt = tf - t #how long until the end of the simulation
		h = min(h, intOPT['max_step']) #whichever is smaller use that
		h = max(h, intOPT['min_step']) 
		if endOfSimulDt <= h and endOfSimulDt <= h:
			stop = True
			h = endOfSimulDt
		#with the step size that is desired known, try to compute the step
		k1q = findCrossTerm(t, y, BField, EField, physOPT, t0, tf, p_old, p_new, v_old, v_new)
		
		temp = qv_mult(rodriguezQuat(k1q, rk45COEF['B21']*h), y)
		k2q = findCrossTerm(t+rk45COEF['A2']*h, temp, BField, EField, physOPT, t0, tf, p_old, p_new, v_old, v_new)
		
		temp = qv_mult(qMult(rodriguezQuat(k2q, rk45COEF['B32']*h), rodriguezQuat(k1q, rk45COEF['B31']*h)), y)
		k3q = findCrossTerm(t+rk45COEF['A3']*h, temp, BField, EField, physOPT, t0, tf, p_old, p_new, v_old, v_new)
		
		temp = qv_mult(qMult(rodriguezQuat(k3q, rk45COEF['B43']*h), qMult(rodriguezQuat(k2q, rk45COEF['B42']*h), rodriguezQuat(k1q, rk45COEF['B41']*h))), y)
		k4q = findCrossTerm(t+rk45COEF['A4']*h, temp, BField, EField, physOPT, t0, tf, p_old, p_new, v_old, v_new)
		
		temp = qv_mult(qMult(rodriguezQuat(k4q, rk45COEF['B54']*h), qMult(rodriguezQuat(k3q, rk45COEF['B53']*h), 
				qMult(rodriguezQuat(k2q, rk45COEF['B52']*h), rodriguezQuat(k1q, rk45COEF['B51']*h)))), y)
		k5q = findCrossTerm(t+rk45COEF['A5']*h, temp, BField, EField, physOPT, t0, tf, p_old, p_new, v_old, v_new)
		
		temp = qv_mult(qMult(rodriguezQuat(k5q, rk45COEF['B65']*h), qMult(rodriguezQuat(k4q, rk45COEF['B64']*h), 
				qMult(rodriguezQuat(k3q, rk45COEF['B63']*h), qMult(rodriguezQuat(k2q, rk45COEF['B62']*h), rodriguezQuat(k1q, rk45COEF['B61']*h))))), y)
		k6q = findCrossTerm(t+rk45COEF['A6']*h, temp, BField, EField, physOPT, t0, tf, p_old, p_new, v_old, v_new)
		weightedStepq = qv_mult(qMult(rodriguezQuat(k6q, h*rk45COEF['CH6']), qMult(rodriguezQuat(k5q, h*rk45COEF['CH5']), 
			qMult(rodriguezQuat(k4q, h*rk45COEF['CH4']), qMult(rodriguezQuat(k3q, h*rk45COEF['CH3']),
			qMult(rodriguezQuat(k2q, h*rk45COEF['CH2']), rodriguezQuat(k1q, h*rk45COEF['CH1'])))))), y)
		
		TE1 = qv_mult(qMult(rodriguezQuat(k5q, h*rk45COEF['C5']), qMult(rodriguezQuat(k4q, h*rk45COEF['C4']), 
							qMult(rodriguezQuat(k3q, h*rk45COEF['C3']), qMult(rodriguezQuat(k2q, h*rk45COEF['C2']), 
								rodriguezQuat(k1q, h*rk45COEF['C1']))))), y)
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
			return particle
	particle['s'] = y[:]
	particle['n_spin_steps'] += nstep
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
	h = np.float64(intOPT['h'])
	coefs = __RK75109Coefs()
	t = t0
	stop = False
	hmin = 1.0E-9
	nstep = 0
	while(1):
		nstep+=1
		endOfSimulDt = tf - t #how long until the end of the simulation
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
	h = np.float64(intOPT['h'])
	coefs = __RK75109Coefs()
	t = t0
	stop = False
	hmin = 1.0E-9
	nstep = 0
	while(1):
		nstep+=1
		endOfSimulDt = tf - t #how long until the end of the simulation
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
			return particle
	particle['s'] = y[:]
	particle['n_spin_steps'] += nstep
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
	h = np.float64(intOPT['h'])
	t = t0
	stop = False
	nstep = 0
	while not stop:
		nstep+=1
		h = np.float64(intOPT['h']) #reset to the base h value at the beginning each time
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
	h = np.float64(intOPT['h'])
	t = t0
	stop = False
	nstep = 0
	while not stop:
		nstep+=1
		h = np.float64(intOPT['h']) #reset to the base h value at the beginning each time 
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
def integrateSpin(particle, simulationOptions, spinOptions, BField, EField):
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
	else:
		None
	return particle