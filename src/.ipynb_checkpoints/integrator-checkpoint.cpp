#include "../include/integrator.h"

#include <unistd.h>
#include <math.h>

#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
#include <hiprand/hiprand.h>
#include <hiprand/hiprand_kernel.h>
#define __PREPROC__ __host__ __device__
#elif defined(__NVCOMPILER) || defined(__NVCC__)
#define __PREPROC__ __host__ __device__
#else
#include <random>
#define __PREPROC__
#endif

using namespace std;

__PREPROC__ double sign(double a, double b)
{
  return (b < 0.0)? -abs(a) : abs(a);
}

__PREPROC__ double min_d(double a, double b)
{
  return (a < b)?a:b;
}

__PREPROC__ double max_d(double a, double b)
{
  return (a > b)?a:b;
}

__PREPROC__ double3 pulse(const double t){
	// return {19.1026874e-6*cos(3000*t), 0.0, 0.0};
	return {0.0, 0.0, 0.0};
}

__PREPROC__ double3 grad(double3& pos){
	return {0.0, 0.0, 0.0};
}

/*

__PREPROC__ void obs_dense(long nr, double xold, double x, double3 y, double3 pos_old, double3 v_old, int* irtrn, options opts, 
		double* lastOutput, unsigned int *lastIndex, outputDtype* outputArray, double hout, double3& rcont1, double3& rcont2, 
        double3& rcont3, double3& rcont4, double3& rcont5, double3& rcont6, double3& rcont7, double3& rcont8){
	////("%ld %lf %lf %lf %lf\n", nr, x, y.x, y.y, y.z);
	////("%lf %lf %lf %lf\n", *lastOutput, xold, x, opts.ioutInt);
	double s;
	double s1;
	double3 dense_out;
	
	while(*lastOutput < x){
		s = (*lastOutput - xold)/hout;
		s1 = 1.0 - s;
		dense_out = rcont1+s*(rcont2+s1*(rcont3+s*(rcont4+s1*(rcont5+s*(rcont6+s1*(rcont7+s*rcont8))))));
		////("\t %ld %lf\n", *lastIndex, *lastOutput);
		////("%d %d %d %d\n", *lastIndex, *lastIndex+1, *lastIndex+2, *lastIndex+3);
		double3 a = (double3){0.0, G_CONST, 0.0};
		double3 outPos = pos_old + v_old * (x-xold) + 0.5*a*(x-xold)*(x-xold);
		outputDtype temp;
		temp.t = *lastOutput;
		temp.x.x = outPos.x;
		temp.x.y = outPos.y;
		temp.x.z = outPos.z;
		temp.s.x = dense_out.x;
		temp.s.y = dense_out.y;
		temp.s.z = dense_out.z;
		outputArray[*lastIndex] = temp;
		*lastIndex += 1;
		*lastOutput += opts.ioutInt;
	}
}

__PREPROC__ void obs(long nr, double xold, double x, double3 y, double3 pos, int* irtrn, options opts, 
		double* lastOutput, unsigned int *lastIndex, outputDtype* outputArray){
	////("%ld %lf %lf %lf %lf\n", nr, x, y.x, y.y, y.z);
	////("%lf %lf %lf %lf\n", *lastOutput, xold, x, opts.ioutInt);
	
	while(*lastOutput < x){
		////("\t %ld %lf\n", *lastIndex, *lastOutput);
		////("%d %d %d %d\n", *lastIndex, *lastIndex+1, *lastIndex+2, *lastIndex+3);
		outputDtype temp;
		temp.t = x;
		temp.x.x = pos.x;
		temp.x.y = pos.y;
		temp.x.z = pos.z;
		temp.s.x = y.x;
		temp.s.y = y.y;
		temp.s.z = y.z;
		outputArray[*lastIndex] = temp;
		*lastOutput += opts.ioutInt;
	}
}
*/

__PREPROC__ void interpolate(const double t, const double t0, const double tf, 
		const double3& p_old, const double3& p_new, const double3& v_old, 
		const double3& v_new, double3& p_out, double3& v_out){
	p_out = (p_old*(tf-t) + p_new*(t-t0))/(tf-t0);
	// //("%f\t %f\t %f\n", p_old[0], p_out[0], p_new[0]);
	v_out = v_old;
}

__PREPROC__ double3 findCrossTerm(const double t, const double3 y, const double3 B0, const double3 E, 
					 const double gamma, const double t0, const double tf, const double3 p_old,
					 const double3 p_new, const double3 v_old, const double3 v_new){
	double3 p, v, G;
	double3 B = pulse(t) + B0;
	interpolate(t,t0,tf,p_old,p_new,v_old,v_new,p,v);
	//G = grad(p);
	B = pulse(t) + B0 + 1.0/c2*cross(v, E);
	return gamma * B;
}

__PREPROC__ void Bloch(const double t, const double3& y, double3& f, const double3 B0, const double3 E,
			const double gamma, const double t0, const double tf , const double3& p_old,
			const double3& p_new, const double3& v_old, const double3& v_new){
	double3 temp;
	temp = findCrossTerm(t, y, B0, E, gamma, t0, tf, p_old, p_new, v_old, v_new);
	f = cross(y, temp);
}

// double hinit(double x, double* y, double posneg, double* f0, double* f1, double* yy1, int iord, options OPT)
// {
//     double dnf, dny, atoli, rtoli, sk, h, h1, der2, der12, sqr;
//     unsigned i;
//     int n = 3;

//     dnf = 0.0;
//     dny = 0.0;
//     atoli = OPT.atol;
//     rtoli = OPT.rtol;

//     for (i = 0; i < n; i++){
//         sk = atoli + rtoli * std::abs(y[i]);
//         sqr = f0[i] / sk;
//         dnf += sqr*sqr;
//         sqr = y[i] / sk;
//         dny += sqr*sqr;
//     }

//     if ((dnf <= 1.0E-10) || (dny <= 1.0E-10))
//     h = 1.0E-6;
//     else
//     h = sqrt (dny/dnf) * 0.01;

//     h = min_d(h, OPT.hmax);
//     h = sign(h, posneg);

//     /* perform an explicit Euler step */
//     for (i = 0; i < n; i++)
//         yy1[i] = y[i] + h * f0[i];
//     Bloch (x+h, yy1, f1);

//     /* estimate the second derivative of the solution */
//     der2 = 0.0;
//     for (i = 0; i < n; i++){
//         sk = atoli + rtoli * std::abs(y[i]);
//         sqr = (f1[i] - f0[i]) / sk;
//         der2 += sqr*sqr;
//     }
//     der2 = sqrt (der2) / h;

//     /* step size is computed such that h**iord * max_d(norm(f0),norm(der2)) = 0.01 */
//     der12 = max_d(std::abs(der2), sqrt(dnf));
//     if (der12 <= 1.0E-15)
//     h1 = max_d (1.0E-6, std::abs(h)*1.0E-3);
//     else
//     h1 = pow (0.01/der12, 1.0/(double)iord);
//     h = min_d (100.0 * std::abs(h), min_d (h1, OPT.hmax));

//     return sign (h, posneg);

// }

__PREPROC__ int integrateDOP(double t0, double tf, double3& y, const double3& p_old, 
		const double3& p_new, const double3& v_old, const double3& v_new, 
		options OPT){

    double3 yy1, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10;
    //double3 rcont1, rcont2, rcont3, rcont4, rcont5, rcont6, rcont7, rcont8;
    //int arret, idid;
    //int iasti, iord, irtrn, reject, last, nonsti;
    int reject, last;
	double facold, expo1, fac, facc1, facc2, fac11, posneg, xph;
    double err2, deno;
	double3 erri, sqr, sk;
    double atoli, rtoli, hlamb, err, hnew;
    unsigned int nfcn = 0, nstep = 0, naccpt = 0, nrejct = 0;
    double x = t0;
    double xf = tf;
    double h = OPT.h;
    int n = 3;

    facold = 1.0E-4;
    expo1 = 1.0/8.0 - OPT.beta * 0.2;
    facc1 = 1.0 / OPT.fac1;
    facc2 = 1.0 / OPT.fac2;
    posneg = sign(1.0, tf-t0);

    /* initial preparations */
    atoli = OPT.atol;
    rtoli = OPT.rtol;
    last  = 0;
    hlamb = 0.0;
	////("k1 prior = %lf %lf %lf\n", k1.x, k1.y, k1.z);
    Bloch(x, y, k1, OPT.B0, OPT.E, OPT.gamma, t0, tf, p_old, p_new, v_old, v_new);
	////("k1 post = %lf %lf %lf\n", k1.x, k1.y, k1.z);

    double hmax = std::abs(OPT.hmax);
    // if (OPT.h == 0.0)
    //     h = hinit(fcn, x0, y, posneg, k1, k2, k3, iord, hmax, OPT.atol, OPT.rtol);
    nfcn += 2;
    reject = 0;
	/*
    if (OPT.iout){
        irtrn = 1;
        hout = 1.0;
        xout = t0;
        obs(naccpt+1, xold, x, y, p_old, &irtrn, OPT);
    }*/

    while (1){

        if (nstep > OPT.nmax){
            return -2;
        }

        if (0.1 * std::abs(h) <= std::abs(x) * OPT.uround){
            return -3;
        }

        if ((x + 1.01*h - xf) * posneg > 0.0){
            h = xf - x;
            last = 1;
        }

        nstep++;

        /* the twelve stages */
		////("yy1 = %lf %lf %lf\n", yy1.x, yy1.y, yy1.z);
        yy1 = y + h * COEF::a21 * k1;
		////("yy12 = %lf %lf %lf\n", yy1.x, yy1.y, yy1.z);

        Bloch(x+COEF::c2*h, yy1, k2, OPT.B0, OPT.E, OPT.gamma, t0, tf, p_old, p_new, v_old, v_new);
		
        yy1 = y + h * (COEF::a31*k1 + COEF::a32*k2);
        Bloch(x+COEF::c3*h, yy1, k3, OPT.B0, OPT.E, OPT.gamma, t0, tf, p_old, p_new, v_old, v_new);

        yy1 = y + h * (COEF::a41*k1 + COEF::a43*k3);
        Bloch(x+COEF::c4*h, yy1, k4, OPT.B0, OPT.E, OPT.gamma, t0, tf, p_old, p_new, v_old, v_new);

        yy1 = y + h * (COEF::a51*k1 + COEF::a53*k3 + COEF::a54*k4);
        Bloch(x+COEF::c5*h, yy1, k5, OPT.B0, OPT.E, OPT.gamma, t0, tf, p_old, p_new, v_old, v_new);

        yy1 = y + h * (COEF::a61*k1 + COEF::a64*k4 + COEF::a65*k5);
        Bloch(x+COEF::c6*h, yy1, k6, OPT.B0, OPT.E, OPT.gamma, t0, tf, p_old, p_new, v_old, v_new);

        yy1 = y + h * (COEF::a71*k1 + COEF::a74*k4 + COEF::a75*k5 + COEF::a76*k6);
        Bloch(x+COEF::c7*h, yy1, k7, OPT.B0, OPT.E, OPT.gamma, t0, tf, p_old, p_new, v_old, v_new);

        yy1 = y + h * (COEF::a81*k1 + COEF::a84*k4 + COEF::a85*k5 + COEF::a86*k6 + COEF::a87*k7);
        Bloch(x+COEF::c8*h, yy1, k8, OPT.B0, OPT.E, OPT.gamma, t0, tf, p_old, p_new, v_old, v_new);

        yy1 = y + h * (COEF::a91*k1 + COEF::a94*k4 + COEF::a95*k5 + COEF::a96*k6 + COEF::a97*k7 + COEF::a98*k8);
        Bloch(x+COEF::c9*h, yy1, k9, OPT.B0, OPT.E, OPT.gamma, t0, tf, p_old, p_new, v_old, v_new);

        yy1 = y + h * (COEF::a101*k1 + COEF::a104*k4 + COEF::a105*k5 + COEF::a106*k6 + COEF::a107*k7 + COEF::a108*k8 + COEF::a109*k9);
        Bloch(x+COEF::c10*h, yy1, k10, OPT.B0, OPT.E, OPT.gamma, t0, tf, p_old, p_new, v_old, v_new);

        yy1 = y + h * (COEF::a111*k1 + COEF::a114*k4 + COEF::a115*k5 + COEF::a116*k6 + COEF::a117*k7 + COEF::a118*k8 + COEF::a119*k9 + COEF::a1110*k10);

        Bloch(x+COEF::c11*h, yy1, k2, OPT.B0, OPT.E, OPT.gamma, t0, tf, p_old, p_new, v_old, v_new);
        xph = x + h;

        yy1 = y + h * (COEF::a121*k1 + COEF::a124*k4 + COEF::a125*k5 + COEF::a126*k6 + COEF::a127*k7 + COEF::a128*k8 + COEF::a129*k9 + COEF::a1210*k10 + COEF::a1211*k2);

        Bloch(xph, yy1, k3, OPT.B0, OPT.E, OPT.gamma, t0, tf, p_old, p_new, v_old, v_new);
        nfcn += 11;

        k4 = COEF::b1*k1 + COEF::b6*k6 + COEF::b7*k7 + COEF::b8*k8 + COEF::b9*k9 + COEF::b10*k10 + COEF::b11*k2 + COEF::b12*k3;
		
        k5 = y + h * k4;
        
        /* error estimation */
        err = 0.0;
        err2 = 0.0;
		sk = atoli + rtoli * max_d3(fabs3(y), fabs3(k5));
		erri = k4 - COEF::bhh1*k1 - COEF::bhh2*k9 - COEF::bhh3*k3;
        sqr = erri / sk;
        err2 += sum(sqr*sqr);
        erri = COEF::er1*k1 + COEF::er6*k6 + COEF::er7*k7 + COEF::er8*k8 + COEF::er9*k9 +
            COEF::er10 * k10 + COEF::er11*k2 + COEF::er12*k3;
        sqr = erri / sk;
        err += sum(sqr*sqr);

        deno = err + 0.01 * err2;
        if (deno <= 0.0)
			deno = 1.0;
        err = std::abs(h) * err * sqrt (1.0 / (deno*(double)n));

        /* computation of hnew */
        fac11 = pow (err, expo1);
        /* Lund-stabilization */
        fac = fac11 / pow(facold,OPT.beta);
        /* we require fac1 <= hnew/h <= fac2 */
        fac = max_d (facc2, min_d (facc1, fac/OPT.safe));
        hnew = h / fac;

        if (err <= 1.0)
        {
            /* step accepted */

            facold = max_d (err, 1.0E-4);
            naccpt++;
            Bloch(xph, k5, k4, OPT.B0, OPT.E, OPT.gamma, t0, tf, p_old, p_new, v_old, v_new);
            nfcn++;
			/*
			// final preparation for dense output
			if (OPT.iout == 2)
			{
				//save the first function evaluations
				rcont1 = y;
				ydiff = k5 - y;
				rcont2 = ydiff;
				bspl = h * k1 - ydiff;
				rcont3 = bspl;
				rcont4 = ydiff - h*k4 - bspl;
				rcont5 = COEF::d41*k1 + COEF::d46*k6 + COEF::d47*k7 + COEF::d48*k8 +
					COEF::d49*k9 + COEF::d410*k10 + COEF::d411*k2 + COEF::d412*k3;
				rcont6 = COEF::d51*k1 + COEF::d56*k6 + COEF::d57*k7 + COEF::d58*k8 +
					COEF::d59*k9 + COEF::d510*k10 + COEF::d511*k2 + COEF::d512*k3;
				rcont7 = COEF::d61*k1 + COEF::d66*k6 + COEF::d67*k7 + COEF::d68*k8 +
					COEF::d69*k9 + COEF::d610*k10 + COEF::d611*k2 + COEF::d612*k3;
				rcont8 = COEF::d71*k1 + COEF::d76*k6 + COEF::d77*k7 + COEF::d78*k8 +
					COEF::d79*k9 + COEF::d710*k10 + COEF::d711*k2 + COEF::d712*k3;

				//the next three function evaluations 
			   yy1 = y + h * (COEF::a141*k1 + COEF::a147*k7 + COEF::a148*k8 +
					COEF::a149*k9 + COEF::a1410*k10 + COEF::a1411*k2 +
					COEF::a1412*k3 + COEF::a1413*k4);
				Bloch(x+COEF::c14*h, yy1, k10, OPT.B0, OPT.E, OPT.gamma, t0, tf, p_old, p_new, v_old, v_new);
				yy1 = y + h * (COEF::a151*k1 + COEF::a156*k6 + COEF::a157*k7 + COEF::a158*k8 +
					COEF::a1511*k2 + COEF::a1512*k3 + COEF::a1513*k4 +
					COEF::a1514*k10);
				Bloch(x+COEF::c15*h, yy1, k2, OPT.B0, OPT.E, OPT.gamma, t0, tf, p_old, p_new, v_old, v_new);
				yy1 = y + h * (COEF::a161*k1 + COEF::a166*k6 + COEF::a167*k7 + COEF::a168*k8 +
							COEF::a169*k9 + COEF::a1613*k4 + COEF::a1614*k10 +
							COEF::a1615*k2);
				Bloch(x+COEF::c16*h, yy1, k3, OPT.B0, OPT.E, OPT.gamma, t0, tf, p_old, p_new, v_old, v_new);
				nfcn += 3;

				//final preparation

				rcont5 = h * (rcont5 + COEF::d413*k4 + COEF::d414*k10 +
						COEF::d415*k2 + COEF::d416*k3);
				rcont6 = h * (rcont6 + COEF::d513*k4 + COEF::d514*k10 +
						COEF::d515*k2 + COEF::d516*k3);
				rcont7 = h * (rcont7 + COEF::d613*k4 + COEF::d614*k10 +
						COEF::d615*k2 + COEF::d616*k3);
				rcont8 = h * (rcont8 + COEF::d713*k4 + COEF::d714*k10 +
						COEF::d715*k2 + COEF::d716*k3);
			}
			*/
			k1 = k4;
			y = k5;
			x = xph;
			/*
			if (OPT.iout == 1){
				hout = h;
				xout = x;
				obs(naccpt+1, xold, x, y, p_old, &irtrn, OPT, &lastOutput, &lastIndex, outputArray);
				if (irtrn < 0)
					return 2;
			} else if (OPT.iout == 2){
				hout = h;
				xout = x;
				obs_dense(naccpt+1, xold, x, y, p_old, v_old, &irtrn, OPT, &lastOutput, &lastIndex, outputArray, hout, rcont1, rcont2, rcont3, rcont4, rcont5, rcont6, rcont7, rcont8);
				if (irtrn < 0)
					return 2;
			}*/

            // normal exit
            if (last)
            {
                return 1;
            }

            if (std::abs(hnew) > hmax)
                hnew = posneg * hmax;
            if (reject)
                hnew = posneg * min_d (std::abs(hnew), std::abs(h));

            reject = 0;
        }
        else{
            /* step rejected */
            hnew = h / min_d (facc1, fac11/OPT.safe);
            reject = 1;
            if (naccpt >= 1)
				nrejct=nrejct + 1;
            last = 0;
        }

        h = hnew;
    }

}

__PREPROC__ int integrateRK45Hybrid(double t0, double tf, double3& y, const double3& p_old, 
		const double3& p_new, const double3& v_old, const double3& v_new, 
		options OPT, double &h){
	double x = t0;
	double xf = tf;
	bool stop = false;
	double hmin = 1.0E-9;
	int nstep = 0;
	double endOfSimulDt;
	
	double3 k1;
	double3 k2;
	double3 k3;
	double3 k4;
	double3 k5;
	double3 k6;
	double3 yy1;
	double3 TE2;
	double3 weightedStep;
	double absErr;
	double hnew;
	double3 a = {0.0, 0.0, 0.0};
	if(OPT.gravity)
		a.y = G_CONST;
	while(1){
		nstep++;
		endOfSimulDt = xf - x; //how long until the end of the simulation
		//nextOutputDt = lastOutput + OPT.ioutInt - x; //how long to the next output time
		//printf("h = %.10f, hmax = %.10f, min = %.10f\n", h, OPT.hmax, hmin);
		h = min(h, OPT.hmax);
		h = max(h, hmin);
		if(h > endOfSimulDt){
			h = endOfSimulDt;
			stop = true;
		}
		else{
			stop = false;
		}
		//printf("%lf %.10f %lf \n", x, h, endOfSimulDt);
		//sleep(1);
		if(h < OPT.swapStepSize){ //in this case we want to use the traditional RK45 methodology
			//printf("using normal\n");
			Bloch(x, yy1, k1, OPT.B0, OPT.E, OPT.gamma, t0, tf, p_old, p_new, v_old, v_new);
			yy1 = y + h*RK45COEF::B21*k1;
			Bloch(x+RK45COEF::A2*h, yy1, k2, OPT.B0, OPT.E,OPT.gamma, t0, tf, p_old, p_new, v_old, v_new);
			yy1 = y + h*RK45COEF::B31*k1 + h*RK45COEF::B32*k2;
			Bloch(x+RK45COEF::A3*h, yy1, k3, OPT.B0, OPT.E, OPT.gamma, t0, tf, p_old, p_new, v_old, v_new);
			yy1 = y + h*RK45COEF::B41*k1 + h*RK45COEF::B42*k2 + h*RK45COEF::B43*k3;
			Bloch(x+RK45COEF::A4*h, yy1, k4, OPT.B0, OPT.E, OPT.gamma, t0, tf, p_old, p_new, v_old, v_new);
			yy1 = y + h*RK45COEF::B51*k1 + h*RK45COEF::B52*k2 + h*RK45COEF::B53*k3 + h*RK45COEF::B54*k4;
			Bloch(x+RK45COEF::A5*h, yy1, k5, OPT.B0, OPT.E, OPT.gamma, t0, tf, p_old, p_new, v_old, v_new);
			yy1 = y + h*RK45COEF::B61*k1 + h*RK45COEF::B62*k2 + h*RK45COEF::B63*k3 + h*RK45COEF::B64*k4 + h*RK45COEF::B65*k5;
			Bloch(x+RK45COEF::A6*h, yy1, k6, OPT.B0, OPT.E, OPT.gamma, t0, tf, p_old, p_new, v_old, v_new);
			weightedStep = y + h*(k1*RK45COEF::CH1+k2*RK45COEF::CH2+k3*RK45COEF::CH3+k4*RK45COEF::CH4+k5*RK45COEF::CH5+k6*RK45COEF::CH6);
			TE2 = h*(RK45COEF::CT1*k1 + RK45COEF::CT2*k2 + RK45COEF::CT3*k3 + RK45COEF::CT4*k4 + RK45COEF::CT5*k5 + RK45COEF::CT6*k6);
			//printf("weightedStep = %lf %lf %lf %lf\n", h, weightedStep.x, weightedStep.y, weightedStep.z);
		}
		else{
			//printf("using rotations\n");
			k1 = findCrossTerm(x, y, OPT.B0, OPT.E, OPT.gamma, t0, tf, p_old, p_new, v_old, v_new);
			quaternion temp2 = rodriguezQuat(k1, RK45COEF::B21*h);
			yy1 = qv_mult(rodriguezQuat(k1, RK45COEF::B21*h), y);
			k2 = findCrossTerm(x+RK45COEF::A2*h, yy1, OPT.B0, OPT.E, OPT.gamma, t0, tf, p_old, p_new, v_old, v_new);
			yy1 = qv_mult(qMult(rodriguezQuat(k2, RK45COEF::B32*h), rodriguezQuat(k1, RK45COEF::B31*h)), y);
			k3 = findCrossTerm(x+RK45COEF::A3*h, yy1, OPT.B0, OPT.E, OPT.gamma, t0, tf, p_old, p_new, v_old, v_new);
			yy1 = qv_mult(qMult(rodriguezQuat(k3, RK45COEF::B43*h), qMult(rodriguezQuat(k2, RK45COEF::B42*h), rodriguezQuat(k1, RK45COEF::B41*h))), y);
			k4 = findCrossTerm(x+RK45COEF::A4*h, yy1, OPT.B0, OPT.E, OPT.gamma, t0, tf, p_old, p_new, v_old, v_new);
			yy1 = qv_mult(qMult(rodriguezQuat(k4, RK45COEF::B54*h), qMult(rodriguezQuat(k3, RK45COEF::B53*h), 
					qMult(rodriguezQuat(k2, RK45COEF::B52*h), rodriguezQuat(k1, RK45COEF::B51*h)))), y);
			k5 = findCrossTerm(x+RK45COEF::A5*h, yy1, OPT.B0, OPT.E, OPT.gamma, t0, tf, p_old, p_new, v_old, v_new);
			yy1 = qv_mult(qMult(rodriguezQuat(k5, RK45COEF::B65*h), qMult(rodriguezQuat(k4, RK45COEF::B64*h), 
					qMult(rodriguezQuat(k3, RK45COEF::B63*h), qMult(rodriguezQuat(k2, RK45COEF::B62*h), rodriguezQuat(k1, RK45COEF::B61*h))))), y);
			k6 = findCrossTerm(x+RK45COEF::A6*h, yy1, OPT.B0, OPT.E, OPT.gamma, t0, tf, p_old, p_new, v_old, v_new);
			weightedStep = qv_mult(qMult(rodriguezQuat(k6, h*RK45COEF::CH6), qMult(rodriguezQuat(k5, h*RK45COEF::CH5), 
				qMult(rodriguezQuat(k4, h*RK45COEF::CH4), qMult(rodriguezQuat(k3, h*RK45COEF::CH3),
				qMult(rodriguezQuat(k2, h*RK45COEF::CH2), rodriguezQuat(k1, h*RK45COEF::CH1)))))), y);
			TE2 = qv_mult(qMult(rodriguezQuat(k5, h*RK45COEF::C5), qMult(rodriguezQuat(k4, h*RK45COEF::C4), 
								qMult(rodriguezQuat(k3, h*RK45COEF::C3), qMult(rodriguezQuat(k2, h*RK45COEF::C2), 
									rodriguezQuat(k1, h*RK45COEF::C1))))), y);
			TE2 = weightedStep - TE2;
		}
		absErr = len(TE2);
		if(absErr <= OPT.rtol){ //accept the step and move on to the next one
			x= x + h; //what time are we at now
			y = weightedStep; //update the spin for the next iteration
			/*
			if(out){
				//printf("%lf %lf %lf\n", weightedStep.x, weightedStep.y, weightedStep.z);
				double3 outPos = p_old + v_old * (x-t0) + 0.5*a*(x-t0)*(x-t0);
				outputDtype temp;
				temp.t = x;
				temp.x.x = outPos.x;
				temp.x.y = outPos.y;
				temp.x.z = outPos.z;
				temp.s.x = y.x;
				temp.s.y = y.y;
				temp.s.z = y.z;
				outputArray[lastIndex] = temp;
				lastIndex += 1;
				lastOutput += OPT.ioutInt;
				out = false; //now reset this so we don't automatically output it again
			}*/
		}
		//now update the time step for the next calculation
		if(absErr < 1.0E-16)
			absErr = 1.0E-16;
		hnew = 0.9 * h * pow(OPT.rtol/absErr, 1.0/5.0);
		h = hnew;
		if(stop){
			//("h = %lf\n", h);
			return nstep;
		}
	}
	return 0;
}
