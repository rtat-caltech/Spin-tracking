#include "../include/particle.h"
#include <unistd.h>
#include <float.h>
#include <stdint.h>

/*
Outputs the sign of a number.
*/

#define UNI_32BIT_INV 2.3283064365386962890625e-10
#define UNI_64BIT_INV 5.42101086242752217003726400434970e-20

#if defined(__HIPCC__)
#define __PREPROCD__ __device__
#elif defined(__NVCOMPILER) || defined(__NVCC__)
#define __PREPROCD__ __device__
#else
#include <random>
#define __PREPROCD__ 
#endif

using namespace std;

template <typename T>
__PREPROCD__ double sgn(const T val) {
	if(val<0)
		return -1.0;
	else
		return 1.0;
	//return (T(0) < val) - (val < T(0)); //we don't want the 0 case
}

__PREPROCD__ uint64_t rol64(const uint64_t x, const int k)
{
	return (x << k) | (x >> (64 - k));
}

/*
//no longer used but leaving it here in case we want to add it back later on
__PREPROCD__ uint64_t particle::splitmix64(uint64_t state) {
	uint64_t result = (state += 0x9E3779B97f4A7C15);
	result = (result ^ (result >> 30)) * 0xBF58476D1CE4E5B9;
	result = (result ^ (result >> 27)) * 0x94D049BB133111EB;
	return result ^ (result >> 31);
}


__PREPROCD__ void particle::xorshift128_init(uint64_t seed) {
    uint64_t splitmix64_state = seed; //apply the seed to that generator
	uint64_t tmp = splitmix64();
	rngState[0] = (uint32_t)tmp;
	rngState[1] = (uint32_t)(tmp >> 32);

	tmp = splitmix64();
	rngState[2] = (uint32_t)tmp;
	rngState[3] = (uint32_t)(tmp >> 32);
}
*/

__PREPROCD__ uint64_t xoshiro256p(rngState& state)
{
   //using this https://en.wikipedia.org/wiki/Xorshift#xoshiro256+
	const uint64_t result = state.x + state.z;
	const uint64_t t = state.y << 17;
	state.z ^= state.x;
	state.w ^= state.y;
	state.y ^= state.z;
	state.x ^= state.w;
	state.z ^= t;
	state.w = rol64(state.z, 45);
	return result;
}

__PREPROCD__ static inline double DoubleFromBits(const uint64_t i){
    return (i >> 11) * 0x1.0p-53;
}

__PREPROCD__ double uniform(rngState& state){
    //A random number between low and high based on the xoshiro256** algorithm
	//https://en.wikipedia.org/wiki/Xorshift#xoshiro256**
    //https://prng.di.unimi.it/
    uint64_t temp = xoshiro256p(state);
    const double out = DoubleFromBits(temp);
    return out;
}

__PREPROCD__ double uniform(rngState& state, const double low, const double high){
    return uniform(state)*(high-low)+low;
}

__PREPROCD__ double normal(rngState& state, const double mean, const double std){
    //Generates a random value from a normal distribution using the Marsaglia Polar Method.
	//https://en.wikipedia.org/wiki/Marsaglia_polar_method
    if(state.hasSpare){
        state.hasSpare = false;
        return state.spare * std + mean;
    }
    else{
        double ts, tu, tv;
        do {
            tu = uniform(state, -1.0, 1.0);
            tv = uniform(state, -1.0, 1.0);
            ts = tu*tu + tv*tv;
        } while (ts >= 1.0 || ts == 0.0);
        ts = sqrt(-2.0*log(ts)/ts);
        state.spare = tv * ts;
        state.hasSpare = true;
        return mean + std * tu * ts;
    }
}

__PREPROCD__ double maxboltz(rngState& state, const double sqrtkT_m){
    return normal(state, 0.0, 1.0) * sqrtkT_m;
}

__PREPROCD__ double unif02pi(rngState& state){
    return uniform(state) * 2.0 * M_PI;
}

__PREPROCD__ double exponential(rngState& state, const double tc){
    return - tc * log(1.0 - uniform(state));
}

__PREPROCD__ void calc_next_collision_time(double t, double tf, double3 v, double3 pos, double& next_gas_coll_time, double& dt, char& coll_type, size_t &n_bounce, size_t &n_coll, bool& finished, char& wall_hit, rngState& state, const options opt) {
    double dx, dy, dz, dtx, dty, dtz = 0.0;
	if(opt.gravity){
		//calculate distance to collision point
		dx = sgn(v.x) * opt.L.x / 2.0 - pos.x;
		dz = sgn(v.z) * opt.L.z / 2.0 - pos.z;
 
		//time to wall for x and z coordinate
		dtx = dx / v.x;
		dtz = dz / v.z;
		double y2 = v.y * v.y;
		if(sgn(v.y) <= 0.0){ //if the particle has negative y velocity
				dy = pos.y + opt.L.y*0.5;
				double sqr = sqrt(-2.0*G_CONST*dy+y2);
				double temp1 = -(sqr+v.y)/G_CONST;
				double temp2 = (sqr-v.y)/G_CONST;
				dty = min(std::abs(temp1), std::abs(temp2));
		}
		else{
				double maxHeight = -0.5 * y2/G_CONST + pos.y;
				if(maxHeight < 0.5 * opt.L.y){ //in this case it can't hit the ceiling
						dy = pos.y+opt.L.y*0.5;
						double sqr = sqrt(-2.0*G_CONST*dy+y2);
						double temp1 = -(sqr+v.y)/G_CONST;
						double temp2 = (sqr-v.y)/G_CONST;
						dty = max(temp1, temp2);
				}
				else{
						dy = opt.L.y*0.5 - pos.y; //how far to ceiling
						double sqr = sqrt(-2.0*G_CONST*dy+y2);
						double temp1 = -(sqr+v.y)/G_CONST;
						double temp2 = (sqr-v.y)/G_CONST;
						dty = min(std::abs(temp1), std::abs(temp2));
				}
		}
		if (dtx < 1e-16 || std::isnan(dtx))
			dtx = 1e6;
		else if (dty < 1e-16 || std::isnan(dty))
			dty = 1e6;
		else if (dtz < 1e-16 || std::isnan(dtz))
			dtz = 1e6;
	}
	else{
		
		dx = sgn(v.x) * opt.L.x / 2.0 - pos.x;
		dy = sgn(v.y) * opt.L.y / 2.0 - pos.y;
		dz = sgn(v.z) * opt.L.z / 2.0 - pos.z;
		
		dtx = dx / v.x;
		dty = dy / v.y;
		dtz = dz / v.z;

		if (dtx < 1e-16)
			dtx = 1e6;
		else if (dty < 1e-16)
			dty = 1e6;
		else if (dtz < 1e-16)
			dtz = 1e6;
	}
	int min_elm;
    double tbounce;
	if(dtx <= dty && dtx <= dtz){
		tbounce = dtx;
		min_elm = 0;
	}
	else if(dty <= dtx && dty <= dtz){
		tbounce = dty;
		min_elm = 1;
	}
	else if(dtz <= dtx && dtz <= dty){
		tbounce = dtz;
		min_elm = 2;
	}
	double timeToNextGas = next_gas_coll_time - t;
	if(opt.maxPosStep <= timeToNextGas && opt.maxPosStep <= tbounce && t + opt.maxPosStep < tf){ //check if the max step size is smaller than the next collision times
		//if so then just say we don't collide and keep going
		dt = opt.maxPosStep;
		coll_type = 'N';
	}
	else if(tbounce < timeToNextGas && t + tbounce < tf){ //is a wall bounce next
		dt = tbounce;
		n_bounce += 1;
		coll_type = 'W';
		if (min_elm == 0)
			wall_hit = 'x';
		else if (min_elm == 1)
			wall_hit = 'y';
		else if (min_elm == 2)
			wall_hit = 'z';
	}
	else if (next_gas_coll_time < tf) { //is a gas collision next?
		dt = next_gas_coll_time - t;
		next_gas_coll_time += exponential(state, opt.tc);
		coll_type = 'G';
		n_coll += 1;
	}
	else { //in this case it reached the end of the simulation
		coll_type = 'N';
		dt = tf - t;
		finished = true;
	}
}

/*
Calculates the new velocities after a wall or gas collision.
*/

__PREPROCD__ void new_velocities(double3 &v, double3 &v_old, char &coll_type, char &wall_hit, rngState &state, const options opt) {
	v_old = v;
    double Vel = len(v);
	if (coll_type == 'N'){
		//in this case we don't have a wall collision and it's just iterating through space still
		//don't update the velocities they're fine
	}
    else if(coll_type == 'W'){
        //if it's a wall collision, check to see if diffuse scattering is on or not
        bool diffuse = false;
        if(opt.diffuse > FLT_MIN){
            //could maybe have diffuse scattering, so sample the RNG to see if it happens
            //printf("%lf %lf %d\n", temp, (double)opt.diffuse, temp < (double)opt.diffuse);
            diffuse = uniform(state) < (double)opt.diffuse;
        }
        if(diffuse){//if we want to do a diffuse collision, do this
            //V = sqrt(vx * vx + vy * vy + vz * vz);
            double phi,theta;
            phi = acos(sqrt(uniform(state)));
            theta = unif02pi(state);
            if (wall_hit == 'x') {
                v.x = -1 * sgn(v.x) * Vel * cos(phi);
                v.y = -Vel * sin(phi) * cos(theta);
                v.z = Vel * sin(phi) * sin(theta);
            }
            else if (wall_hit == 'y') {
                v.x = Vel * sin(phi) * cos(theta);
                v.y = -1 * sgn(v.y) * Vel * cos(phi);
                v.z = Vel * sin(phi) * sin(theta);
            }
            else if (wall_hit == 'z') {
                v.x = Vel * sin(phi) * cos(theta);
                v.y = Vel * sin(phi) * sin(theta);
                v.z = -1 * sgn(v.z) * Vel * cos(phi);
            } 
        }
        else{//otehrwise just flip the velocities around and call it a day
            if (wall_hit == 'x')
                v.x *= -1.0;
            else if (wall_hit == 'y')
                v.y *= -1.0;
            else if (wall_hit == 'z')
                v.z *= -1.0;
        }
    }
    else if(coll_type == 'G'){
        //in this case it's a "gas"/phonon collision
        if (opt.dist == 'M'){
            v.x = maxboltz(state, opt.sqrtKT_m);
            v.y = maxboltz(state, opt.sqrtKT_m);
            v.z = maxboltz(state, opt.sqrtKT_m);
        }
        else if(opt.dist == 'C') {
            double3 vec;
            vec.x = normal(state, 0.0, 1.0);
            vec.y = normal(state, 0.0, 1.0);
            vec.z = normal(state, 0.0, 1.0);
            double vec_norm = len(vec);
            v = Vel * vec/vec_norm;
        }
	}
}

/*
Moves the particle based on the particle velocity and calcuated timestep.
*/
__PREPROCD__ void move(double &t_old, double& t, double3 &pos_old, double3 &pos, double3& v, double3& v_old, double &dt, const options opt) {
	t_old = t; // update the time
	t += dt; //increment forward
	pos_old = pos; //update old position
	v_old = v; //update old velocity
	double3 a;
	if(opt.gravity)
		a = (double3){0.0, G_CONST, 0.0}; //acceleration due to gravity
	else
		a = (double3){0.0, 0.0, 0.0};
	pos = pos_old +  v * dt + 0.5 * a * dt * dt; //update position
	v = v + a * dt; //update velocity
	
}

/*
Convenience function that calls the step() function repeatedL.y until the end time is reached.
*/

#if defined(__HIPCC__) || defined(__NVCOMPILER) || defined(__NVCC__)
__global__ void initParticlesGPU(options opt, double3 *S, double3 *v, double3 *v_old,
                              double3 *pos, double3 *pos_old, double *t, double *t_old,
                              double *tf, double *dt, double *next_gas_coll_time, double *h,
                              rngState *state, size_t *n_bounce, size_t *n_coll, size_t *n_steps,
                              unsigned int *partID, bool *stopParticle, char *coll_type, char *wall_hit){
    unsigned int ipart = threadIdx.x + blockIdx.x * blockDim.x;
    if(ipart < opt.numParticles){
        rngState rng;
        uint64_t tempstate = opt.seed + ipart;
        uint64_t result = (tempstate += 0x9E3779B97f4A7C15);
        result = (result ^ (result >> 30)) * 0xBF58476D1CE4E5B9;
        result = (result ^ (result >> 27)) * 0x94D049BB133111EB;
        result = result ^ (result >> 31);
        rng.x = (uint32_t)result;
        rng.y = (uint32_t)(result >> 32);
        result = (tempstate += 0x9E3779B97f4A7C15);
        result = (result ^ (result >> 30)) * 0xBF58476D1CE4E5B9;
        result = (result ^ (result >> 27)) * 0x94D049BB133111EB;
        result = result ^ (result >> 31);
        rng.z = (uint32_t)result;
        rng.w = (uint32_t)(result >> 32);
        uniform(rng);//scramble the state a few more times just to get things really going
        uniform(rng);//yes this is highly recommended because otherwise the positions will be strongly correlated
        uniform(rng);
        
        //now handle the position and velocity information
        double3 tempos;
        tempos.x = uniform(rng)*opt.L.x-opt.L.x/2.0;
        tempos.y = uniform(rng)*opt.L.y-opt.L.y/2.0;
        tempos.z = uniform(rng)*opt.L.z-opt.L.z/2.0;
        
        pos[ipart] = tempos;
        pos_old[ipart] = tempos;
        
        double3 tempv;
        if (opt.gas_coll == true){
            next_gas_coll_time[ipart] = exponential(rng, opt.tc);
        }
        else
            next_gas_coll_time[ipart] = DBL_MAX; //basically set to the end of time
        if (opt.dist == 'C') {
            if(std::abs(opt.V) <= 1.0e-6){
                //assume the user meant 0 speed at this point
                tempv = (double3){0.0, 0.0, 0.0};
            }
            else{
                tempv.x = normal(rng, 0.0, 1.0);
                tempv.y = normal(rng, 0.0, 1.0);
                tempv.z = normal(rng, 0.0, 1.0);
                double vec_norm = len(tempv);
                tempv = opt.V * tempv/vec_norm;
            }
        }
        else if (opt.dist == 'M') {
            tempv.x = maxboltz(rng, opt.sqrtKT_m);
            tempv.y = maxboltz(rng, opt.sqrtKT_m);
            tempv.z = maxboltz(rng, opt.sqrtKT_m);
        }

        v[ipart] = tempv;
        v_old[ipart] = tempv;
        state[ipart] = rng; //save the final rng state after all this stuff
        //initialize the rest of the particle data to standard values
        S[ipart] = opt.yi;
        t[ipart] = opt.t0;
        t_old[ipart] = opt.t0;
        tf[ipart] = opt.tf;
        dt[ipart] = opt.h;
        h[ipart] = opt.h;
        stopParticle[ipart] = false;
        coll_type[ipart] = 'W';       
    }
}

__global__ void runSimulationGPU(options opt, double3 *pS, double3 *pv, double3 *pv_old,
                              double3 *ppos, double3 *ppos_old, double *pt, double *pt_old,
                              double *ptf, double *pdt, double *pnext_gas_coll_time, double *ph,
                              rngState *pstate, size_t *pn_bounce, size_t *pn_coll, size_t *pn_steps,
                              unsigned int *ppartID, bool *pstopParticle, char *pcoll_type, char *pwall_hit,
                              double nextTOut){
    unsigned int ipart = threadIdx.x + blockIdx.x * blockDim.x;
    if(ipart < opt.numParticles){
        //load the individual particle data
        double3 S = pS[ipart];
        double3 v = pv[ipart];
        double3 v_old = pv_old[ipart];
        double3 pos = ppos[ipart];
        double3 pos_old = ppos_old[ipart];
        
        double t = pt[ipart];
        double t_old = pt_old[ipart];
        double tf; //= ptf[ipart]; // no need to load it's done below
        double dt; //= pdt[ipart]; //no need to load it's defined below
        double next_gas_coll_time = pnext_gas_coll_time[ipart];
        double h = ph[ipart];
        
        rngState state = pstate[ipart];
        size_t n_bounce = pn_bounce[ipart];
        size_t n_coll = pn_coll[ipart];
        size_t n_steps = pn_steps[ipart];
        unsigned int partID = ppartID[ipart];
        bool stopParticle = pstopParticle[ipart];
        char coll_type = pcoll_type[ipart];
        char wall_hit = pwall_hit[ipart];
        dt = nextTOut - t;
        tf = nextTOut;
        bool finished = false;
        int spinResult = 0;
        double tempH = h;
        //now start the actual integration and tracking process
        while(finished == false && stopParticle == false){
            calc_next_collision_time(t, tf, v, pos, next_gas_coll_time, dt, coll_type, n_bounce, n_coll, finished, wall_hit, state, opt);
            move(t_old, t, pos_old, pos, v, v_old, dt, opt);
            new_velocities(v, v_old, coll_type, wall_hit, state, opt);
            //printf("t_old = %lf, t = %lf S = %lf %lf %lf\n", t_old, t, S.x, S.y, S.z);
            //printf("pos_old = %lf %lf %lf, pos = %lf %lf %lf\n", pos_old, pos);
            //printf("v_old = %lf %lf %lf, v = %lf %lf %lf\n", pos_old, pos);
            if(opt.integratorType == 0){
                //use the DOP853 algorithm for spin tracking
                spinResult = integrateDOP(t_old, t, S, pos_old, pos, v_old, v, opt, tempH);
            }
            else if(opt.integratorType == 1){
                //use the default RK45 method, no quaternions or anything
                spinResult = integrateRK45(t_old, t, S, pos_old, pos, v_old, v, opt, tempH);
            }
            else if(opt.integratorType == 2){
                //use the MagnusCFET method
                spinResult = integrateMagnusCFET(t_old, t, S, pos_old, pos, v_old, v, opt, tempH);
            }
            else if(opt.integratorType == 3){
                //rk45 method but with quaternions instead
                spinResult = integrateRK45Quaternion(t_old, t, S, pos_old, pos, v_old, v, opt, tempH);
            }
            else if(opt.integratorType == 4){
                //different set of coefficients for RK45
                spinResult = integrateRKF45(t_old, t, S, pos_old, pos, v_old, v, opt, tempH);
            }
            else if(opt.integratorType == 5){
                //same as option 3 but for option 4's coefficients
                spinResult = integrateRKF45Quaternion(t_old, t, S, pos_old, pos, v_old, v, opt, tempH);
            }
            else{
                //do nothing
                spinResult = 0;
            }      
            if(opt.keepStepSize)
                h = tempH;
            if (spinResult < 0){
                printf("particle %d: error state detected %d\n", ipart, spinResult);
                stopParticle = true;
                t = nan("");
                pos = (double3){nan(""), nan(""), nan("")};
                v = (double3){nan(""), nan(""), nan("")};
                S = (double3){nan(""), nan(""), nan("")};
            }
            n_steps += 1;
        }
        pS[ipart] = S;
        pv[ipart] = v;
        pv_old[ipart] = v_old;
        ppos[ipart] = pos;
        ppos_old[ipart] = pos_old;
        pt[ipart] = t;
        pt_old[ipart] = t_old;
        pnext_gas_coll_time[ipart] = next_gas_coll_time;
        ph[ipart] = h;
        pstate[ipart] = state;
        pn_bounce[ipart] = n_bounce;
        pn_coll[ipart] = n_coll;
        pn_steps[ipart] = n_steps;
        ppartID[ipart] = partID;
        pstopParticle[ipart] = stopParticle;
        pcoll_type[ipart] = coll_type;
        pwall_hit[ipart] = wall_hit;
        ptf[ipart] = tf;
    }
}

#else
void initParticlesCPU(options opt, double3 *S, double3 *v, double3 *v_old,
                              double3 *pos, double3 *pos_old, double *t, double *t_old,
                              double *tf, double *dt, double *next_gas_coll_time, double *h,
                              rngState *state, size_t *n_bounce, size_t *n_coll, size_t *n_steps,
                              unsigned int *partID, bool *stopParticle, char *coll_type, char *wall_hit){
    #if defined(_OPENMP)
    #pragma omp parallel for
    #endif
    for(int ipart = 0; ipart < opt.numParticles; ipart++){
        rngState rng;
        uint64_t tempstate = opt.seed + ipart;
        uint64_t result = (tempstate += 0x9E3779B97f4A7C15);
        result = (result ^ (result >> 30)) * 0xBF58476D1CE4E5B9;
        result = (result ^ (result >> 27)) * 0x94D049BB133111EB;
        result = result ^ (result >> 31);
        rng.x = (uint32_t)result;
        rng.y = (uint32_t)(result >> 32);
        result = (tempstate += 0x9E3779B97f4A7C15);
        result = (result ^ (result >> 30)) * 0xBF58476D1CE4E5B9;
        result = (result ^ (result >> 27)) * 0x94D049BB133111EB;
        result = result ^ (result >> 31);
        rng.z = (uint32_t)result;
        rng.w = (uint32_t)(result >> 32);
        uniform(rng);//scramble the state a few more times just to get things really going
        uniform(rng);//yes this is highly recommended because otherwise the positions will be strongly correlated
        uniform(rng);


        //now handle the position and velocity information
        double3 tempos;
        tempos.x = uniform(rng)*opt.L.x-opt.L.x/2.0;
        tempos.y = uniform(rng)*opt.L.y-opt.L.y/2.0;
        tempos.z = uniform(rng)*opt.L.z-opt.L.z/2.0;
        pos[ipart] = tempos;
        pos_old[ipart] = tempos;
        if (opt.gas_coll == true){
            next_gas_coll_time[ipart] = exponential(rng, opt.tc);
        }
        else
            next_gas_coll_time[ipart] = DBL_MAX; //basically set to the end of time
        if (opt.dist == 'C') {
            double3 vec;
            vec.x = normal(rng, 0.0, 1.0);
            vec.y = normal(rng, 0.0, 1.0);
            vec.z = normal(rng, 0.0, 1.0);

            double vec_norm = len(vec);
            v[ipart] = opt.V * vec/vec_norm;
        }
        else if (opt.dist == 'M') {
            double3 vec;
            vec.x = maxboltz(rng, opt.sqrtKT_m);
            vec.y = maxboltz(rng, opt.sqrtKT_m);
            vec.z = maxboltz(rng, opt.sqrtKT_m);
            v[ipart] = vec;
        }
        v_old[ipart] = v[ipart];
        state[ipart] = rng; //save the final rng state after all this stuff
        //initialize the rest of the particle data to standard values
        S[ipart] = opt.yi;
        t[ipart] = opt.t0;
        t_old[ipart] = opt.t0;
        tf[ipart] = opt.tf;
        dt[ipart] = opt.h;
        h[ipart] = opt.h;
        stopParticle[ipart] = false;
        coll_type[ipart] = 'W';
    }
}

void runSimulationCPU(options opt, double3 *S, double3 *v, double3 *v_old,
                              double3 *pos, double3 *pos_old, double *t, double *t_old,
                              double *tf, double *dt, double *next_gas_coll_time, double *h,
                              rngState *state, size_t *n_bounce, size_t *n_coll, size_t *n_steps,
                              unsigned int *partID, bool *stopParticle, char *coll_type, char *wall_hit,
                              double nextTOut){
    #if defined(_OPENMP)
    #pragma omp parallel for
    #endif
    for(unsigned int ipart = 0; ipart < opt.numParticles; ipart++){
        //load the individual particle data
        double3 S = S[ipart];
        double3 v = v[ipart];
        double3 v_old = v_old[ipart];
        double3 pos = pos[ipart];
        double3 pos_old = pos_old[ipart];
        double t = t[ipart];
        double t_old = t_old[ipart];
        double tf; //= tf[ipart]; // no need to load it's done below
        double dt; //= dt[ipart]; //no need to load it's defined below
        double next_gas_coll_time = next_gas_coll_time[ipart];
        double h = h[ipart];
        rngState state = state[ipart];
        size_t n_bounce = n_bounce[ipart];
        size_t n_coll = n_coll[ipart];
        size_t n_steps = n_steps[ipart];
        unsigned int partID = partID[ipart];
        bool stopParticle = stopParticle[ipart];
        char coll_type = coll_type[ipart];
        char wall_hit = wall_hit[ipart];
        dt = nextTOut - tf[ipart];
        tf = nextTOut;
        bool finished = false;
        int spinResult = 0;
        double tempH = h;
        //now start the actual integration and tracking process
        while(finished == false && stopParticle == false){
            calc_next_collision_time(t, tf, v, pos, next_gas_coll_time, dt, coll_type, n_bounce, n_coll, finished, wall_hit, state, opt);
            move(t_old, t, pos_old, pos, v, v_old, dt, opt);
            new_velocities(v, v_old, coll_type, wall_hit, state, opt);
            if(opt.integratorType == 0){
                //use the DOP853 algorithm for spin tracking
                spinResult = integrateDOP(t_old, t, S, pos_old, pos, v_old, v, opt, tempH);
            }
            else if(opt.integratorType == 1){
                //use the default RK45 method, no quaternions or anything
                spinResult = integrateRK45(t_old, t, S, pos_old, pos, v_old, v, opt, tempH);
            }
            else if(opt.integratorType == 2){
                //use the MagnusCFET method
                spinResult = integrateMagnusCFET(t_old, t, S, pos_old, pos, v_old, v, opt, tempH);
            }
            else if(opt.integratorType == 3){
                //rk45 method but with quaternions instead
                spinResult = integrateRK45Quaternion(t_old, t, S, pos_old, pos, v_old, v, opt, tempH);
            }
            else if{opt.integratorType == 4){
                //different set of coefficients for RK45
                spinResult = integrateRKF45(t_old, t, S, pos_old, pos, v_old, v, opt, tempH);
            }
            else if{opt.integratorType == 5){
                //same as option 3 but for option 4's coefficients
                spinResult = integrateRKF45Quaternion(t_old, t, S, pos_old, pos, v_old, v, opt, tempH);
            }
            else{
                //do nothing
                spinResult = 0;
            }      
            if(opt.keepStepSize)
                h = tempH;
            if (spinResult < 0){
                stopParticle = true;
                t = nan("");
                pos = (double3){nan(""), nan(""), nan("")};
                v = (double3){nan(""), nan(""), nan("")};
                S = (double3){nan(""), nan(""), nan("")};
            }
            n_steps += 1;

        }
    }
}
#endif