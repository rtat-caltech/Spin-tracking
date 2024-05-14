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

__PREPROC__ _PREC sign(_PREC a, _PREC b)
{
  return (b < 0.0)? -abs(a) : abs(a);
}

__PREPROC__ _PREC min_d(_PREC a, _PREC b)
{
  return (a < b)?a:b;
}

__PREPROC__ _PREC max_d(_PREC a, _PREC b)
{
  return (a > b)?a:b;
}

// Extra pulse used for testing purposes
__PREPROC__ coords testNoise(const _PREC t, coords a, coords w) {
	return a * ((coords) {sin(w.x * t), sin(w.y * t), sin(w.z * t)});
}

__PREPROC__ coords pulse(const _PREC t, _PREC a, _PREC w){
	return {0.0, 0.0, a * cos(w*t)};
}

__PREPROC__ coords grad(coords& pos, const options OPT){
	return {dot(pos, OPT.Gx), dot(pos, OPT.Gy), dot(pos, OPT.Gz)};
}

__PREPROC__ void interpolate(const _PREC t, const _PREC t0, const _PREC tf, 
		const coords& p_old, const coords& p_new, const coords& v_old, 
		const coords& v_new, coords& p_out, coords& v_out, const options OPT){
    if(OPT.gravity){
        const coords a = {0.0, G_CONST, 0.0};
        p_out = p_old + v_old * (t-t0) + 0.5 * a * (t-t0)*(t-t0);
        v_out = v_old + a * (t-t0);
        
    }
    else{
        p_out = p_old + v_old * (t-t0);
        v_out = v_old;
    }
}

__PREPROC__ coords findCrossTerm(const _PREC t, const options OPT, const _PREC t0, const _PREC tf, const coords p_old,
					 const coords p_new, const coords v_old, const coords v_new){
	coords p, v, G, B, N;
	interpolate(t,t0,tf,p_old,p_new,v_old,v_new,p,v,OPT);
	G = grad(p, OPT);
	B = pulse(t, OPT.a, OPT.w) + OPT.B0 + 1.0/c2*cross(v, OPT.E) + G;
	return OPT.gamma * B;
}

__PREPROC__ coords findNoiseTerm(const _PREC t, const options OPT, const _PREC t0, const _PREC tf, const coords p_old,
					 const coords p_new, const coords v_old, const coords v_new){
	coords p, v, G, B, N;
	interpolate(t,t0,tf,p_old,p_new,v_old,v_new,p,v,OPT);
	G = grad(p, OPT);
	N = testNoise(t, OPT.noiseAmplitudes, OPT.noiseFrequencies);
	B = 1.0/c2*cross(v, OPT.E) + G + N;
	return OPT.gamma * B;
}


__PREPROC__ void Bloch(const _PREC t, const coords& y, coords& f, const options OPT, 
			const _PREC t0, const _PREC tf , const coords& p_old,
			const coords& p_new, const coords& v_old, const coords& v_new){
	const coords temp = findCrossTerm(t, OPT, t0, tf, p_old, p_new, v_old, v_new);
	f = cross(y, temp);
}

__PREPROC__ int integrateDOP(const _PREC t0, const _PREC tf, coords& y, const coords& p_old, 
		const coords& p_new, const coords& v_old, const coords& v_new, 
		const options OPT, _PREC &h){
    coords yy1, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10;
    //int arret, idid;
    //int iasti, iord, irtrn, reject, last, nonsti;
    int reject, last;
    _PREC facold, expo1, fac, facc1, facc2, fac11, posneg, xph;
    _PREC err2, deno;
    coords erri, sqr, sk;
    _PREC err, hnew;
    unsigned int nfcn = 0, nstep = 0, naccpt = 0, nrejct = 0;
    _PREC x = t0;
    _PREC xf = tf;
    int n = 3;
    facold = 1.0E-4;
    expo1 = 1.0/8.0 - OPT.beta * 0.2;
    facc1 = 1.0 / OPT.fac1;
    facc2 = 1.0 / OPT.fac2;
    posneg = sign(1.0, tf-t0);

    /* initial preparations */
    last  = 0;
    ////("k1 prior = %lf %lf %lf\n", k1.x, k1.y, k1.z);
    Bloch(x, y, k1, OPT, t0, tf, p_old, p_new, v_old, v_new);
    ////("k1 post = %lf %lf %lf\n", k1.x, k1.y, k1.z);
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
            return -1;
        }
        /*
        if (0.1 * std::abs(h) <= std::abs(x) * OPT.uround){
            return -3;
        }
        */
        if ((x + 1.01*h - xf) * posneg > 0.0){
            h = xf - x;
            last = 1;
        }
        if (h < OPT.hmin && last != 1){
            h = OPT.hmin;
        }
        nstep++;

        /* the twelve stages */
        //printf("h = %0.17f, yy1 = %lf %lf %lf\n", h, yy1.x, yy1.y, yy1.z);
        yy1 = y + h * COEF::a21 * k1;
        ////("yy12 = %lf %lf %lf\n", yy1.x, yy1.y, yy1.z);
        Bloch(x+COEF::c2*h, yy1, k2, OPT, t0, tf, p_old, p_new, v_old, v_new);

        yy1 = y + h * (COEF::a31*k1 + COEF::a32*k2);
        Bloch(x+COEF::c3*h, yy1, k3, OPT, t0, tf, p_old, p_new, v_old, v_new);

        yy1 = y + h * (COEF::a41*k1 + COEF::a43*k3);
        Bloch(x+COEF::c4*h, yy1, k4, OPT, t0, tf, p_old, p_new, v_old, v_new);

        yy1 = y + h * (COEF::a51*k1 + COEF::a53*k3 + COEF::a54*k4);
        Bloch(x+COEF::c5*h, yy1, k5, OPT, t0, tf, p_old, p_new, v_old, v_new);

        yy1 = y + h * (COEF::a61*k1 + COEF::a64*k4 + COEF::a65*k5);
        Bloch(x+COEF::c6*h, yy1, k6, OPT, t0, tf, p_old, p_new, v_old, v_new);

        yy1 = y + h * (COEF::a71*k1 + COEF::a74*k4 + COEF::a75*k5 + COEF::a76*k6);
        Bloch(x+COEF::c7*h, yy1, k7, OPT, t0, tf, p_old, p_new, v_old, v_new);

        yy1 = y + h * (COEF::a81*k1 + COEF::a84*k4 + COEF::a85*k5 + COEF::a86*k6 + COEF::a87*k7);
        Bloch(x+COEF::c8*h, yy1, k8, OPT, t0, tf, p_old, p_new, v_old, v_new);

        yy1 = y + h * (COEF::a91*k1 + COEF::a94*k4 + COEF::a95*k5 + COEF::a96*k6 + COEF::a97*k7 + COEF::a98*k8);
        Bloch(x+COEF::c9*h, yy1, k9, OPT, t0, tf, p_old, p_new, v_old, v_new);

        yy1 = y + h * (COEF::a101*k1 + COEF::a104*k4 + COEF::a105*k5 + COEF::a106*k6 + COEF::a107*k7 + COEF::a108*k8 + COEF::a109*k9);
        Bloch(x+COEF::c10*h, yy1, k10, OPT, t0, tf, p_old, p_new, v_old, v_new);

        yy1 = y + h * (COEF::a111*k1 + COEF::a114*k4 + COEF::a115*k5 + COEF::a116*k6 + COEF::a117*k7 + COEF::a118*k8 + COEF::a119*k9 + COEF::a1110*k10);

        Bloch(x+COEF::c11*h, yy1, k2, OPT, t0, tf, p_old, p_new, v_old, v_new);
        xph = x + h;

        yy1 = y + h * (COEF::a121*k1 + COEF::a124*k4 + COEF::a125*k5 + COEF::a126*k6 + COEF::a127*k7 + COEF::a128*k8 + COEF::a129*k9 + COEF::a1210*k10 + COEF::a1211*k2);

        Bloch(xph, yy1, k3, OPT, t0, tf, p_old, p_new, v_old, v_new);
        nfcn += 11;

        k4 = COEF::b1*k1 + COEF::b6*k6 + COEF::b7*k7 + COEF::b8*k8 + COEF::b9*k9 + COEF::b10*k10 + COEF::b11*k2 + COEF::b12*k3;
		
        k5 = y + h * k4;
        
        /* error estimation */
        err = 0.0;
        err2 = 0.0;
		sk = OPT.atol + OPT.rtol * max_d3(fabs3(y), fabs3(k5));
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
        err = std::abs(h) * err * sqrt (1.0 / (deno*(_PREC)n));

        /* computation of hnew */
        fac11 = pow (err, expo1);
        /* Lund-stabilization */
        fac = fac11 / pow(facold,OPT.beta);
        /* we require fac1 <= hnew/h <= fac2 */
        fac = max_d (facc2, min_d (facc1, fac/OPT.safe));
        hnew = h / fac;
        if (err <= 1.0 || OPT.fixedStepSize){ //do this operation if we want to save the step, so in the case of adaptive if the error is low
            //in the case of fixed step size always do it
            /* step accepted */
            facold = max_d (err, 1.0E-4);
            naccpt++;
            Bloch(xph, k5, k4, OPT, t0, tf, p_old, p_new, v_old, v_new);
            nfcn++;
			k1 = k4;
			y = k5;
			x = xph;
			
            // normal exit
            if (last)
            {
                return 1;
            }

            if (std::abs(hnew) > OPT.hmax)
                hnew = posneg * OPT.hmax;
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
        if(!OPT.fixedStepSize)//only update the step size if we are doing adaptive
            h = hnew;
    }

}

__PREPROC__ int integrateRK45(const _PREC t0, const _PREC tf, coords& y, const coords& p_old, 
		const coords& p_new, const coords& v_old, const coords& v_new, 
		const options OPT, _PREC &h){
    //traditional RK45 integrator
	_PREC t = t0;
	bool stop = false;
	int nstep = 0;
	_PREC endOfSimulDt;
	_PREC lastH = h;
	coords k1, k2, k3, k4, k5, k6, yy1, TE2, weightedStep;
	_PREC error, tol, ratio, q;
    _PREC beta1 = 0.7;
	_PREC beta2 = -0.4;
	_PREC accept_safety = 0.81;
	_PREC k = 7.0;
	_PREC prev_ratio = 1.0;
	while(1){
		nstep++;
        if (nstep > OPT.nmax){
            //if we have taken too many steps, we should stop the code
            return -2;
        }
		endOfSimulDt = tf - t; //how long until the end of the simulation
        lastH = h; //update our last time step that we took before we calculate the new step size
        if(OPT.fixedStepSize){
            //if using a fixed step size use that value
            h = OPT.h;
        }
        else{
            //otherwise make sure h is in the valid range
            //only throw an error if it takes too small of a step because that can kill the code
            if(h > OPT.hmax) {
                h = min(h, OPT.hmax);			
            } else if(h < OPT.hmin) {
                return -1;
			}
        }
		if(h >= endOfSimulDt){
            //now check if h is too large for the amount of time left, if so make it the right size
			h = endOfSimulDt;
			stop = true; //tell the system to stop
		}
		else{
            //otherwise we still think the system should be running and keep on going
            lastH = h;
            stop = false;
		}
        Bloch(t, yy1, k1, OPT, t0, tf, p_old, p_new, v_old, v_new);
        yy1 = y + h*RK45COEF::B21*k1;
        Bloch(t+RK45COEF::A2*h, yy1, k2, OPT, t0, tf, p_old, p_new, v_old, v_new);
        yy1 = y + h*RK45COEF::B31*k1 + h*RK45COEF::B32*k2;
        Bloch(t+RK45COEF::A3*h, yy1, k3, OPT, t0, tf, p_old, p_new, v_old, v_new);
        yy1 = y + h*RK45COEF::B41*k1 + h*RK45COEF::B42*k2 + h*RK45COEF::B43*k3;
        Bloch(t+RK45COEF::A4*h, yy1, k4, OPT, t0, tf, p_old, p_new, v_old, v_new);
        yy1 = y + h*RK45COEF::B51*k1 + h*RK45COEF::B52*k2 + h*RK45COEF::B53*k3 + h*RK45COEF::B54*k4;
        Bloch(t+RK45COEF::A5*h, yy1, k5, OPT, t0, tf, p_old, p_new, v_old, v_new);
        yy1 = y + h*RK45COEF::B61*k1 + h*RK45COEF::B62*k2 + h*RK45COEF::B63*k3 + h*RK45COEF::B64*k4 + h*RK45COEF::B65*k5;
        Bloch(t+RK45COEF::A6*h, yy1, k6, OPT, t0, tf, p_old, p_new, v_old, v_new);
        weightedStep = y + h*(k1*RK45COEF::CH1+k2*RK45COEF::CH2+k3*RK45COEF::CH3+k4*RK45COEF::CH4+k5*RK45COEF::CH5+k6*RK45COEF::CH6);
        TE2 = h*(RK45COEF::CT1*k1 + RK45COEF::CT2*k2 + RK45COEF::CT3*k3 + RK45COEF::CT4*k4 + RK45COEF::CT5*k5 + RK45COEF::CT6*k6);
		error = len(TE2);
		error = max(error, 1.0E-16); //do this to prevent the step size from collapsing
		tol = OPT.rtol; // TODO: incorporate abs and rel tols
		ratio = tol/error;

		q = pow(ratio, beta1/k) * pow(prev_ratio, beta2/k);
		q = min(q,4.0); // control stepsize growth
		if (q > accept_safety && error < 2 * tol || OPT.fixedStepSize) {
			//in this case the step is accepted, or we're doing fixed step sizes anyways
			y = weightedStep;
			t += h;
			if(stop){
                h = lastH;
				return 0;
			}
		}
		else{
			if(stop){ //in this case we wanted to output but the step wasn't accepted so try again
				stop = false;
			}
		}
        if(!OPT.fixedStepSize)
            h = h*q;
		prev_ratio = ratio;
	}
	return 0;
}

__PREPROC__ int integrateRKF45(const _PREC t0, const _PREC tf, coords& y, const coords& p_old, 
		const coords& p_new, const coords& v_old, const coords& v_new, 
		const options OPT, _PREC &h){
    //traditional RK45 integrator
	_PREC t = t0;
	bool stop = false;
	int nstep = 0;
	_PREC endOfSimulDt;
	_PREC lastH = h;
	coords k1, k2, k3, k4, k5, k6, yy1, TE2, weightedStep;
	_PREC error, tol, ratio, q;
    _PREC beta1 = 0.7;
	_PREC beta2 = -0.4;
	_PREC accept_safety = 0.81;
	_PREC k = 7.0;
	_PREC prev_ratio = 1.0;
	while(1){
		nstep++;
        if (nstep > OPT.nmax){
            //if we have taken too many steps, we should stop the code
            return -2;
        }
		endOfSimulDt = tf - t; //how long until the end of the simulation
        lastH = h; //update our last time step that we took before we calculate the new step size
        if(OPT.fixedStepSize){
            //if using a fixed step size use that value
            h = OPT.h;
        }
        else{
            //otherwise make sure h is in the valid range
            //now check if h is too large for the amount of time left, if so make it the right size
            if(h >= endOfSimulDt){
                //end the simulation
                h = endOfSimulDt;
                stop = true; //tell the system to stop
            }
            else{
                //otherwise we still think the system should be running and keep on going
                lastH = h;
                stop = false;
                //make sure the new h value is allowed then
                if(h > OPT.hmax) {
                    h = min(h, OPT.hmax);
                } else if(h < OPT.hmin) {
                    return -1;
				}
            }
        }
        Bloch(t, yy1, k1, OPT, t0, tf, p_old, p_new, v_old, v_new);
        yy1 = y + h*RKF45COEF::B21*k1;
        Bloch(t+RKF45COEF::A2*h, yy1, k2, OPT, t0, tf, p_old, p_new, v_old, v_new);
        yy1 = y + h*RKF45COEF::B31*k1 + h*RKF45COEF::B32*k2;
        Bloch(t+RKF45COEF::A3*h, yy1, k3, OPT, t0, tf, p_old, p_new, v_old, v_new);
        yy1 = y + h*RKF45COEF::B41*k1 + h*RKF45COEF::B42*k2 + h*RKF45COEF::B43*k3;
        Bloch(t+RKF45COEF::A4*h, yy1, k4, OPT, t0, tf, p_old, p_new, v_old, v_new);
        yy1 = y + h*RKF45COEF::B51*k1 + h*RKF45COEF::B52*k2 + h*RKF45COEF::B53*k3 + h*RKF45COEF::B54*k4;
        Bloch(t+RKF45COEF::A5*h, yy1, k5, OPT, t0, tf, p_old, p_new, v_old, v_new);
        yy1 = y + h*RKF45COEF::B61*k1 + h*RKF45COEF::B62*k2 + h*RKF45COEF::B63*k3 + h*RKF45COEF::B64*k4 + h*RKF45COEF::B65*k5;
        Bloch(t+RKF45COEF::A6*h, yy1, k6, OPT, t0, tf, p_old, p_new, v_old, v_new);
        weightedStep = y + h*(k1*RKF45COEF::C1+k2*RKF45COEF::C2+k3*RKF45COEF::C3+k4*RKF45COEF::C4+k5*RKF45COEF::C5+k6*RKF45COEF::C6);
        TE2 = h*(RKF45COEF::CR1*k1 + RKF45COEF::CR2*k2 + RKF45COEF::CR3*k3 + RKF45COEF::CR4*k4 + RKF45COEF::CR5*k5 + RKF45COEF::CR6*k6);
		error = len(TE2);
		error = max(error, 1.0E-16); //do this to prevent the step size from collapsing
		tol = OPT.rtol; // TODO: incorporate abs and rel tols
		ratio = tol/error;

		q = pow(ratio, beta1/k) * pow(prev_ratio, beta2/k);
		q = min(q,4.0); // control stepsize growth
		if (q > accept_safety && error < 2 * tol || OPT.fixedStepSize) {
			//in this case the step is accepted, or we're doing fixed step sizes anyways
			y = weightedStep;
			t += h;
			if(stop){
                h = lastH;
				return 0;
			}
		}
		else{
			if(stop){ //in this case we wanted to output but the step wasn't accepted so try again
				stop = false;
			}
		}
        if(!OPT.fixedStepSize)
            h = h*q;
		prev_ratio = ratio;
	}
	return 0;
}

__PREPROC__ int integrateRK45Quaternion(const _PREC t0, const _PREC tf, coords& y, const coords& p_old, 
		const coords& p_new, const coords& v_old, const coords& v_new, 
		const options OPT, _PREC &h){
    //traditional RK45 integrator
	_PREC t = t0;
	bool stop = false;
	int nstep = 0;
	_PREC endOfSimulDt;
	_PREC lastH = h;
	coords k1, k2, k3, k4, k5, k6, yy1, TE2, weightedStep;
	_PREC error, tol, ratio, q;
    _PREC beta1 = 0.7;
	_PREC beta2 = -0.4;
	_PREC accept_safety = 0.81;
	_PREC k = 7.0;
	_PREC prev_ratio = 1.0;
	while(1){
		nstep++;
        if (nstep > OPT.nmax){
            //if we have taken too many steps, we should stop the code
            return -2;
        }
		endOfSimulDt = tf - t; //how long until the end of the simulation
        lastH = h; //update our last time step that we took before we calculate the new step size
        if(OPT.fixedStepSize){
            //if using a fixed step size use that value
            h = OPT.h;
        }
        else{
            //otherwise make sure h is in the valid range
            //now check if h is too large for the amount of time left, if so make it the right size
            if(h >= endOfSimulDt){
                //end the simulation
                h = endOfSimulDt;
                stop = true; //tell the system to stop
            }
            else{
                //otherwise we still think the system should be running and keep on going
                lastH = h;
                stop = false;
                //make sure the new h value is allowed then
                if(h > OPT.hmax) {
                    h = min(h, OPT.hmax);
                } else if(h < OPT.hmin) {
                    return -1;
				}
            }
        }
        k1 = findCrossTerm(t, OPT, t0, tf, p_old, p_new, v_old, v_new);
        //yy1 = qv_mult(rodriguezQuat(k1, RK45COEF::B21*h), y);
        k2 = findCrossTerm(t+RK45COEF::A2*h, OPT, t0, tf, p_old, p_new, v_old, v_new);
        //yy1 = qv_mult(qMult(rodriguezQuat(k2, RK45COEF::B32*h), rodriguezQuat(k1, RK45COEF::B31*h)), y);
        k3 = findCrossTerm(t+RK45COEF::A3*h, OPT, t0, tf, p_old, p_new, v_old, v_new);
        //yy1 = qv_mult(qMult(rodriguezQuat(k3, RK45COEF::B43*h), qMult(rodriguezQuat(k2, RK45COEF::B42*h), rodriguezQuat(k1, RK45COEF::B41*h))), y);
        k4 = findCrossTerm(t+RK45COEF::A4*h, OPT, t0, tf, p_old, p_new, v_old, v_new);
        //yy1 = qv_mult(qMult(rodriguezQuat(k4, RK45COEF::B54*h), qMult(rodriguezQuat(k3, RK45COEF::B53*h), 
                //qMult(rodriguezQuat(k2, RK45COEF::B52*h), rodriguezQuat(k1, RK45COEF::B51*h)))), y);
        k5 = findCrossTerm(t+RK45COEF::A5*h, OPT, t0, tf, p_old, p_new, v_old, v_new);
        //yy1 = qv_mult(qMult(rodriguezQuat(k5, RK45COEF::B65*h), qMult(rodriguezQuat(k4, RK45COEF::B64*h), 
                //qMult(rodriguezQuat(k3, RK45COEF::B63*h), qMult(rodriguezQuat(k2, RK45COEF::B62*h), rodriguezQuat(k1, RK45COEF::B61*h))))), y);
        k6 = findCrossTerm(t+RK45COEF::A6*h, OPT, t0, tf, p_old, p_new, v_old, v_new);
        weightedStep = qv_mult(qMult(rodriguezQuat(k6, h*RK45COEF::CH6), qMult(rodriguezQuat(k5, h*RK45COEF::CH5), 
            qMult(rodriguezQuat(k4, h*RK45COEF::CH4), qMult(rodriguezQuat(k3, h*RK45COEF::CH3),
            qMult(rodriguezQuat(k2, h*RK45COEF::CH2), rodriguezQuat(k1, h*RK45COEF::CH1)))))), y);
        TE2 = qv_mult(qMult(rodriguezQuat(k5, h*RK45COEF::C5), qMult(rodriguezQuat(k4, h*RK45COEF::C4), 
                            qMult(rodriguezQuat(k3, h*RK45COEF::C3), qMult(rodriguezQuat(k2, h*RK45COEF::C2), 
                                rodriguezQuat(k1, h*RK45COEF::C1))))), y);
        TE2 = weightedStep - TE2;
		error = len(TE2);
		error = max(error, 1.0E-16); //do this to prevent the step size from collapsing
		tol = OPT.rtol; // TODO: incorporate abs and rel tols
		ratio = tol/error;

		q = pow(ratio, beta1/k) * pow(prev_ratio, beta2/k);
		q = min(q,4.0); // control stepsize growth
		if (q > accept_safety && error < 2 * tol || OPT.fixedStepSize) {
			//in this case the step is accepted, or we're doing fixed step sizes anyways
			y = weightedStep;
			t += h;
			if(stop){
                h = lastH;
				return 0;
			}
		}
		else{
			if(stop){ //in this case we wanted to output but the step wasn't accepted so try again
				stop = false;
			}
		}
        if(!OPT.fixedStepSize)
            h = h*q;
		prev_ratio = ratio;
	}
	return 0;
}

__PREPROC__ int integrateRKF45Quaternion(const _PREC t0, const _PREC tf, coords& y, const coords& p_old, 
		const coords& p_new, const coords& v_old, const coords& v_new, 
		const options OPT, _PREC &h){
    //traditional RK45 integrator
	_PREC t = t0;
	bool stop = false;
	int nstep = 0;
	_PREC endOfSimulDt;
	_PREC lastH = h;
	coords k1, k2, k3, k4, k5, k6, yy1, TE2, weightedStep;
	_PREC error, tol, ratio, q;
    _PREC beta1 = 0.7;
	_PREC beta2 = -0.4;
	_PREC accept_safety = 0.81;
	_PREC k = 7.0;
	_PREC prev_ratio = 1.0;
	while(1){
		nstep++;
        if (nstep > OPT.nmax){
            //if we have taken too many steps, we should stop the code
            return -2;
        }
		endOfSimulDt = tf - t; //how long until the end of the simulation
        lastH = h; //update our last time step that we took before we calculate the new step size
        if(OPT.fixedStepSize){
            //if using a fixed step size use that value
            h = OPT.h;
        }
        else{
            //otherwise make sure h is in the valid range
            //now check if h is too large for the amount of time left, if so make it the right size
            if(h >= endOfSimulDt){
                //end the simulation
                h = endOfSimulDt;
                stop = true; //tell the system to stop
            }
            else{
                //otherwise we still think the system should be running and keep on going
                lastH = h;
                stop = false;
                //make sure the new h value is allowed then
                if(h > OPT.hmax) {
                    h = min(h, OPT.hmax);
				} else if(h < OPT.hmin) {
                    return -1;
				}
            }
        }
        k1 = findCrossTerm(t, OPT, t0, tf, p_old, p_new, v_old, v_new);
        //yy1 = qv_mult(rodriguezQuat(k1, RKF45COEF::B21*h), y);
        k2 = findCrossTerm(t+RKF45COEF::A2*h, OPT, t0, tf, p_old, p_new, v_old, v_new);
        //yy1 = qv_mult(qMult(rodriguezQuat(k2, RKF45COEF::B32*h), rodriguezQuat(k1, RKF45COEF::B31*h)), y);
        k3 = findCrossTerm(t+RKF45COEF::A3*h, OPT, t0, tf, p_old, p_new, v_old, v_new);
        //yy1 = qv_mult(qMult(rodriguezQuat(k3, RKF45COEF::B43*h), qMult(rodriguezQuat(k2, RKF45COEF::B42*h), rodriguezQuat(k1, RKF45COEF::B41*h))), y);
        k4 = findCrossTerm(t+RKF45COEF::A4*h, OPT, t0, tf, p_old, p_new, v_old, v_new);
        //yy1 = qv_mult(qMult(rodriguezQuat(k4, RKF45COEF::B54*h), qMult(rodriguezQuat(k3, RKF45COEF::B53*h), 
                //qMult(rodriguezQuat(k2, RKF45COEF::B52*h), rodriguezQuat(k1, RKF45COEF::B51*h)))), y);
        k5 = findCrossTerm(t+RKF45COEF::A5*h, OPT, t0, tf, p_old, p_new, v_old, v_new);
        //yy1 = qv_mult(qMult(rodriguezQuat(k5, RKF45COEF::B65*h), qMult(rodriguezQuat(k4, RKF45COEF::B64*h), 
                //qMult(rodriguezQuat(k3, RKF45COEF::B63*h), qMult(rodriguezQuat(k2, RKF45COEF::B62*h), rodriguezQuat(k1, RKF45COEF::B61*h))))), y);
        k6 = findCrossTerm(t+RKF45COEF::A6*h, OPT, t0, tf, p_old, p_new, v_old, v_new);
        weightedStep = qv_mult(qMult(rodriguezQuat(k6, h*RKF45COEF::C6), qMult(rodriguezQuat(k5, h*RKF45COEF::C5), 
            qMult(rodriguezQuat(k4, h*RKF45COEF::C4), qMult(rodriguezQuat(k3, h*RKF45COEF::C3),
            qMult(rodriguezQuat(k2, h*RKF45COEF::C2), rodriguezQuat(k1, h*RKF45COEF::C1)))))), y);
        TE2 = qv_mult(qMult(rodriguezQuat(k5, h*RKF45COEF::C5), qMult(rodriguezQuat(k4, h*RKF45COEF::C4), 
                            qMult(rodriguezQuat(k3, h*RKF45COEF::C3), qMult(rodriguezQuat(k2, h*RKF45COEF::C2), 
                                rodriguezQuat(k1, h*RKF45COEF::C1))))), y);
        TE2 = weightedStep - TE2;
		error = len(TE2);
		error = max(error, 1.0E-16); //do this to prevent the step size from collapsing
		tol = OPT.rtol; // TODO: incorporate abs and rel tols
		ratio = tol/error;

		q = pow(ratio, beta1/k) * pow(prev_ratio, beta2/k);
		q = min(q,4.0); // control stepsize growth
		if (q > accept_safety && error < 2 * tol || OPT.fixedStepSize) {
			//in this case the step is accepted, or we're doing fixed step sizes anyways
			y = weightedStep;
			t += h;
			if(stop){
                h = lastH;
				return 0;
			}
		}
		else{
			if(stop){ //in this case we wanted to output but the step wasn't accepted so try again
				stop = false;
			}
		}
        if(!OPT.fixedStepSize)
            h = h*q;
		prev_ratio = ratio;
	}
	return 0;
}

__PREPROC__ int integrateMagnusCFET(const _PREC t0, const _PREC tf, coords& y, const coords& p_old,
						const coords& p_new, const coords& v_old, const coords& v_new, const options OPT, _PREC& h){
	// An implementation of the 8-th order scheme from https://arxiv.org/pdf/1102.5071.pdf
	_PREC t = t0;
	bool stop = false;
	coords k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11;
	coords B1, B2, B3, B4, B5;
	coords y8, y6;
	_PREC beta1 = 0.7;
	_PREC beta2 = -0.4;
	_PREC accept_safety = 0.81;
	_PREC k = 7.0;
	_PREC prev_ratio = 1.0;
	_PREC endOfSimulDt = 0.0;
	_PREC q, tol, error, ratio;
	unsigned int nstep = 0;
    _PREC lastH = h; //store the last iteration value of h
	while (1){
		nstep++;
        if (nstep > OPT.nmax){
            return -2;
        }
		endOfSimulDt = tf - t; //how long until the end of the simulation
        lastH = h;
        if(OPT.fixedStepSize){
            //if using a fixed step size use that value
            h = OPT.h;
        }
        else{
            //otherwise make sure h is in the valid range
            //now check if h is too large for the amount of time left, if so make it the right size
            if(h >= endOfSimulDt){
                //end the simulation
                h = endOfSimulDt;
                stop = true; //tell the system to stop
            }
            else{
                //otherwise we still think the system should be running and keep on going
                lastH = h;
                stop = false;
                //make sure the new h value is allowed then
                if(h > OPT.hmax) {
                    h = min(h, OPT.hmax);
				} else if(h < OPT.hmin) {
                    return -1;
				}
            }
        }
		B1 = findCrossTerm(t+GL5::X1*h, OPT, t0, tf, p_old, p_new, v_old, v_new);
		B2 = findCrossTerm(t+GL5::X2*h, OPT, t0, tf, p_old, p_new, v_old, v_new);
		B3 = findCrossTerm(t+GL5::X3*h, OPT, t0, tf, p_old, p_new, v_old, v_new);
		B4 = findCrossTerm(t+GL5::X4*h, OPT, t0, tf, p_old, p_new, v_old, v_new);
		B5 = findCrossTerm(t+GL5::X5*h, OPT, t0, tf, p_old, p_new, v_old, v_new);

		k11 = (CFET8::G15 * B1 + CFET8::G14 * B2 + CFET8::G13 * B3 + CFET8::G12 * B4 + CFET8::G11 * B5) * h;
		k10 = (CFET8::G25 * B1 + CFET8::G24 * B2 + CFET8::G23 * B3 + CFET8::G22 * B4 + CFET8::G21 * B5) * h;
		k9 = (CFET8::G35 * B1 + CFET8::G34 * B2 + CFET8::G33 * B3 + CFET8::G32 * B4 + CFET8::G31 * B5) * h;
		k8 = (CFET8::G45 * B1 + CFET8::G44 * B2 + CFET8::G43 * B3 + CFET8::G42 * B4 + CFET8::G41 * B5) * h;
		k7 = (CFET8::G55 * B1 + CFET8::G54 * B2 + CFET8::G53 * B3 + CFET8::G52 * B4 + CFET8::G51 * B5) * h;
		k6 = (CFET8::G61 * B1 + CFET8::G62 * B2 + CFET8::G63 * B3 + CFET8::G64 * B4 + CFET8::G65 * B5) * h;
		k5 = (CFET8::G51 * B1 + CFET8::G52 * B2 + CFET8::G53 * B3 + CFET8::G54 * B4 + CFET8::G55 * B5) * h;
		k4 = (CFET8::G41 * B1 + CFET8::G42 * B2 + CFET8::G43 * B3 + CFET8::G44 * B4 + CFET8::G45 * B5) * h;
		k3 = (CFET8::G31 * B1 + CFET8::G32 * B2 + CFET8::G33 * B3 + CFET8::G34 * B4 + CFET8::G35 * B5) * h;
		k2 = (CFET8::G21 * B1 + CFET8::G22 * B2 + CFET8::G23 * B3 + CFET8::G24 * B4 + CFET8::G25 * B5) * h;
		k1 = (CFET8::G11 * B1 + CFET8::G12 * B2 + CFET8::G13 * B3 + CFET8::G14 * B4 + CFET8::G15 * B5) * h;

		y8 = rodriguez(k1, rodriguez(k2, rodriguez(k3, rodriguez(k4, rodriguez(k5, rodriguez(k6, rodriguez(k7, rodriguez(k8, rodriguez(k9, rodriguez(k10, rodriguez(k11, y)))))))))));

		k5 = (CFET6::H15 * B1 + CFET6::H14 * B2 + CFET6::H13 * B3 + CFET6::H12 * B4 + CFET6::H11 * B5) * h;
		k4 = (CFET6::H25 * B1 + CFET6::H24 * B2 + CFET6::H23 * B3 + CFET6::H22 * B4 + CFET6::H21 * B5) * h;
		k3 = (CFET6::H31 * B1 + CFET6::H32 * B2 + CFET6::H33 * B3 + CFET6::H34 * B4 + CFET6::H35 * B5) * h;
		k2 = (CFET6::H21 * B1 + CFET6::H22 * B2 + CFET6::H23 * B3 + CFET6::H24 * B4 + CFET6::H25 * B5) * h;
		k1 = (CFET6::H11 * B1 + CFET6::H12 * B2 + CFET6::H13 * B3 + CFET6::H14 * B4 + CFET6::H15 * B5) * h;

		y6 = rodriguez(k1, rodriguez(k2, rodriguez(k3, rodriguez(k4, rodriguez(k5, y)))));

		error = len(y8 - y6);
        error = max(error, 1.0E-16); //do this to prevent the step size from collapsing
		tol = OPT.rtol; // TODO: incorporate abs and rel tols
		ratio = tol/error;

		q = pow(ratio, beta1/k) * pow(prev_ratio, beta2/k);
		q = min(q,4.0); // control stepsize growth
		if (q > accept_safety && error < 2 * tol || OPT.fixedStepSize) {
			//in this case the step is accepted, or we're doing fixed step sizes anyways
			y = y8;
			t += h;
			if(stop){
                h = lastH;
				return 0;
			}
		}
		else{
			if(stop){ //in this case we wanted to output but the step wasn't accepted so try again
				stop = false;
			}
		}
        if(!OPT.fixedStepSize)
            h = h*q;
		prev_ratio = ratio;
	}
	return 0;
}

__PREPROC__ _PREC first_sample_point(_PREC t0, _PREC h) {
	// Returns the first Spectrum sample point for a time interval starting at t0.
	// With sampling interval h
	return (floor(t0/h)+1)*h; // The smallest multiple of h greater than t0
}

__PREPROC__ int integrateSpectrum(_PREC t0, _PREC tf, SpectrumAggregator& specagg, const coords& p_old, const coords& p_new, const coords& v_old, const coords& v_new, options OPT, const _PREC h) {
	// I'm doing it this way because I'm worried about floating point error
	_PREC t = first_sample_point(t0, h);
	_PREC next_t = first_sample_point(tf, h);
	int n_steps = 0;
	while (t < next_t - (h/2)) {
		coords B = findNoiseTerm(t, OPT, t0, tf, p_old, p_new, v_old, v_new);
		specagg.update(B/2);
		t += h;
		n_steps++;
	}
	
	return n_steps;
}

#if defined(__HIPCC__) || defined(__NVCOMPILER) || defined(__NVCC__)
__PREPROC__ int collectNoiseSamples(_PREC t0, _PREC tf, cufftReal* Bnoise, const coords& p_old,
									const coords& p_new, const coords& v_old, const coords& v_new, options OPT, _PREC h, int integrator_steps) {
	_PREC t = first_sample_point(t0, h);
	_PREC next_t = first_sample_point(tf, h);
	int tlen = timeSeriesLength(OPT.ioutInt, h, true);
	int n_steps = 0;
	while (t < next_t - (h/2)) {
		coords B = findNoiseTerm(t, OPT, t0, tf, p_old, p_new, v_old, v_new)/2;
		int time_index = integrator_steps + n_steps;
		Bnoise[tlen * 0 + time_index] = B.x;
		Bnoise[tlen * 1 + time_index] = B.y;
		Bnoise[tlen * 2 + time_index] = B.z;
		t += h;
		n_steps++;
	}	
	return n_steps;
}
#endif

__PREPROC__ int integrateHamiltonian(_PREC t0, _PREC tf, quaternion& y, options OPT, _PREC h) {
	_PREC t = t0;
	quaternion q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11;
	coords B1, B2, B3, B4, B5;
	coords dummy = {0.0, 0.0, 0.0};
	coords p_old = dummy;
	coords p_new = dummy;
	coords v_old = dummy;
	coords v_new = dummy;

	_PREC endOfSimulDt = 0.0;
	unsigned int nstep = 0;
	while (t < tf){
		nstep++;
        if (nstep > OPT.nmax){
            return -2;
        }
		endOfSimulDt = tf - t;
		if(h >= endOfSimulDt){
			h = endOfSimulDt;
		}
		
		B1 = findCrossTerm(t+GL5::X1*h, OPT, t0, tf, p_old, p_new, v_old, v_new);
		B2 = findCrossTerm(t+GL5::X2*h, OPT, t0, tf, p_old, p_new, v_old, v_new);
		B3 = findCrossTerm(t+GL5::X3*h, OPT, t0, tf, p_old, p_new, v_old, v_new);
		B4 = findCrossTerm(t+GL5::X4*h, OPT, t0, tf, p_old, p_new, v_old, v_new);
		B5 = findCrossTerm(t+GL5::X5*h, OPT, t0, tf, p_old, p_new, v_old, v_new);

		q11 = rodriguezQuat((CFET8::G15 * B1 + CFET8::G14 * B2 + CFET8::G13 * B3 + CFET8::G12 * B4 + CFET8::G11 * B5), h);
		q10 = rodriguezQuat((CFET8::G25 * B1 + CFET8::G24 * B2 + CFET8::G23 * B3 + CFET8::G22 * B4 + CFET8::G21 * B5), h);
		q9 = rodriguezQuat((CFET8::G35 * B1 + CFET8::G34 * B2 + CFET8::G33 * B3 + CFET8::G32 * B4 + CFET8::G31 * B5), h);
		q8 = rodriguezQuat((CFET8::G45 * B1 + CFET8::G44 * B2 + CFET8::G43 * B3 + CFET8::G42 * B4 + CFET8::G41 * B5), h);
		q7 = rodriguezQuat((CFET8::G55 * B1 + CFET8::G54 * B2 + CFET8::G53 * B3 + CFET8::G52 * B4 + CFET8::G51 * B5), h);
		q6 = rodriguezQuat((CFET8::G61 * B1 + CFET8::G62 * B2 + CFET8::G63 * B3 + CFET8::G64 * B4 + CFET8::G65 * B5), h);
		q5 = rodriguezQuat((CFET8::G51 * B1 + CFET8::G52 * B2 + CFET8::G53 * B3 + CFET8::G54 * B4 + CFET8::G55 * B5), h);
		q4 = rodriguezQuat((CFET8::G41 * B1 + CFET8::G42 * B2 + CFET8::G43 * B3 + CFET8::G44 * B4 + CFET8::G45 * B5), h);
		q3 = rodriguezQuat((CFET8::G31 * B1 + CFET8::G32 * B2 + CFET8::G33 * B3 + CFET8::G34 * B4 + CFET8::G35 * B5), h);
		q2 = rodriguezQuat((CFET8::G21 * B1 + CFET8::G22 * B2 + CFET8::G23 * B3 + CFET8::G24 * B4 + CFET8::G25 * B5), h);
		q1 = rodriguezQuat((CFET8::G11 * B1 + CFET8::G12 * B2 + CFET8::G13 * B3 + CFET8::G14 * B4 + CFET8::G15 * B5), h);

		y = q1 * q2 * q3 * q4 * q5 * q6 * q7 * q8 * q9 * q10 * q11 * y;
		t += h;			
	}
	return 0;
}

Matrix2cd integrateFloquetMarkov(_PREC t0, _PREC tf,  Matrix2cd rho, const complex<_PREC> (&Zeta)[2][2], const complex<_PREC> (&Omicron)[2][2]) {
	//_PREC diagonal_decay = -(A(0, 0) + A(1, 1));
	_PREC dt = tf - t0;
	Vector2cd p_diag_0;
	p_diag_0 << rho(0, 0), rho(1, 1);
	Matrix2cd A_diag {
		{-Zeta[1][0]+Zeta[0][1], Zeta[0][1]+Zeta[1][0]},
		{Zeta[1][0]+Zeta[0][1], -Zeta[1][0]+Zeta[0][1]},
	};

	Matrix2cd A_exp = (A_diag * dt).exp();
	Vector2cd p_diag_1 = A_exp * p_diag_0;
	rho(0, 0) = p_diag_1(0);
	rho(1, 1) = p_diag_1(1);
	complex<_PREC> decay_01 = -(Zeta[0][0] + Zeta[0][1] + Zeta[0][1] + Zeta[1][1]);
	complex<_PREC> decay_10 = -(Zeta[1][0] + Zeta[1][1] + Zeta[0][0] + Zeta[1][0]);
	
	Matrix2cd A_off_diag {
		{Omicron[0][1] + decay_01, Omicron[1][0]},
		{Omicron[0][1], Omicron[1][0] + decay_10},
	};
	Matrix2cd A_off_diag_exp = (A_off_diag * dt).exp();
	Vector2cd p_off_diag_0;
	p_off_diag_0 << rho(0, 1), rho(1, 0);
	Vector2cd p_off_diag_1 = A_off_diag_exp * p_off_diag_0;
	rho(0, 1) = p_off_diag_1(0);
	rho(1, 0) = p_off_diag_1(1);
	
	//rho(0, 1) = rho(0, 1) * exp(decay_01 * dt);
	//rho(1, 0) = rho(1, 0) * exp(decay_10 * dt);
	return rho;
}

floquetDiagonalization floquet_diagonalize(options OPT) {
	double t0 = 0.0;
	double tf = (2*M_PI)/OPT.w; //TODO
	int n_prop = 100; // TODO
	vector<quaternion> propagators;
	quaternion y = {1, 0, 0, 0};
	double h = 1.0e-6; // Integration stepsize
	for (int i=0; i < n_prop; i++) {
		double t1 = t0 + (tf - t0) * i/n_prop;
		double t2 = t0 + (tf - t0) * (i+1)/n_prop;
		integrateHamiltonian(t1, t2, y, OPT, h);
		propagators.push_back(y);
	}
	
	quaternion eigen_values = qEigenval(propagators[n_prop-1]);
	quaternion eigen_vectors = qEigenvec(propagators[n_prop-1]);

	double ea = abs(atan2(eigen_values.z, eigen_values.w))/(tf - t0);
	double eb = -ea;
	double deltaE = ea - eb;
	double frequencies[NW];
	int count = 0;
	for(int k=0; k <= NK/2; k++) {
		for (int i=-1; i < 2; i++) {
			double w = deltaE * i + k * OPT.w;
			int index = k*3 + i;
			if (index >= 0) {
				frequencies[index] = w;
				count++;
			}
		}
	}
	floquetDiagonalization fd;
	fd.propagators = propagators;
	fd.f_modes_0 = eigen_vectors;
	fd.f_energies = eigen_values;
	for(int i = 0; i < NW; i++) {
		fd.frequencies[i] = frequencies[i];
	}
	fd.period = tf - t0;
	fd.n_prop = n_prop;
	return fd;

}

coords floquet_integrate(floquetDiagonalization fd, CovarianceSpectrum cspec, options opt) {
	cspec.normalize();
	double Delta[2][2][NK] = {{{0}}};
	complex<double> X[2][2][NK] = {{{0}}};
	complex<double> Gamma[2][2][NK] = {{{0}}};
	complex<double> Zeta[2][2] = {{0}};
	complex<double> Omicron[2][2] = {{0}};
  
	vector<pair<quaternion, Spectrum>> specs = cspec.extract();
	Matrix2cd rho = bloch_to_density(opt.yi, fd.f_modes_0);
	for (int i = 0; i < specs.size(); i++) {
		quaternion c_op = specs.at(i).first;
		Spectrum spec = specs.at(i).second;
		floquet_master_equation_rates(fd, c_op, spec, 
		                              Delta, X, Gamma, Zeta, Omicron);
	}
	rho = integrateFloquetMarkov(opt.t0, opt.tf, rho, Zeta, Omicron);
	int n_period = round((opt.tf - opt.t0)/fd.period);
	return density_to_bloch(rho, fd.f_modes_0 * pow(fd.f_energies, n_period));
}
