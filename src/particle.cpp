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
__PREPROCD__ double particle::sgn(T val) {
	if(val<0)
		return -1.0;
	else
		return 1.0;
	//return (T(0) < val) - (val < T(0)); //we don't want the 0 case
}

__PREPROCD__ uint64_t particle::rol64(uint64_t x, int k)
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

__PREPROCD__ uint64_t particle::xoshiro256p()
{
   //using this https://en.wikipedia.org/wiki/Xorshift#xoshiro256+
	const uint64_t result = rngState[0] + rngState[3];
	const uint64_t t = rngState[1] << 17;
	rngState[2] ^= rngState[0];
	rngState[3] ^= rngState[1];
	rngState[1] ^= rngState[2];
	rngState[0] ^= rngState[3];
	rngState[2] ^= t;
	rngState[3] = rol64(rngState[3], 45);
	return result;
}

__PREPROCD__ static inline double DoubleFromBits(const uint64_t i){
    return (i >> 11) * 0x1.0p-53;
}

__PREPROCD__ double particle::uniform(){
    //A random number between low and high based on the xoshiro256** algorithm
	//https://en.wikipedia.org/wiki/Xorshift#xoshiro256**
    //https://prng.di.unimi.it/
    uint64_t temp = xoshiro256p();
    double out = DoubleFromBits(temp);
    return out;
}

__PREPROCD__ double particle::uniform(double low, double high){
    return uniform()*(high-low)+low;
}

__PREPROCD__ double particle::normal(double mean, double std){
    //Generates a random value from a normal distribution using the Marsaglia Polar Method.
	//https://en.wikipedia.org/wiki/Marsaglia_polar_method
    if(hasSpare){
        hasSpare = false;
        return spareRng * std + mean;
    }
    else{
        double ts, tu, tv;
        do {
            tu = uniform(-1.0, 1.0);
            tv = uniform(-1.0, 1.0);
            ts = tu*tu + tv*tv;
        } while (ts >= 1.0 || ts == 0.0);
        ts = sqrt(-2.0*log(ts)/ts);
        spareRng = tv * ts;
        hasSpare = true;
        return mean + std * tu * ts;
    }
}

__PREPROCD__ double particle::maxboltz(const double sqrtkT_m){
    return normal(0.0, 1.0) * sqrtkT_m;
}

__PREPROCD__ double particle::unif02pi(){
    return uniform() * 2.0 * M_PI;
}

__PREPROCD__ double particle::exponential(const double tc){
    return - tc * log(1.0 - uniform());
}

__PREPROCD__ void particle::calc_next_collision_time(options opt) {
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
	else if (t + tbounce > next_gas_coll_time && next_gas_coll_time < tf) { //is a gas collision next?
		dt = next_gas_coll_time - t;
		next_gas_coll_time += exponential(opt.tc);
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

__PREPROCD__ void particle::new_velocities(options opt) {
	v_old = v;
    double Vel = len(v);
	if (coll_type == 'N'){
		//in this case we don't have a wall collision and it's just iterating through space still
		//don't update the velocities they're fine
	}
	else if (coll_type == 'W' && opt.diffuse == false) {
		if (wall_hit == 'x')
			v.x *= -1.0;
		else if (wall_hit == 'y')
			v.y *= -1.0;
		else if (wall_hit == 'z')
			v.z *= -1.0;
	}
	else if (coll_type == 'W' && opt.diffuse == true) {
		//V = sqrt(vx * vx + vy * vy + vz * vz);
        double phi,theta;
		phi = acos(sqrt(uniform()));
		theta = unif02pi();
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
	else if (coll_type == 'G' && opt.dist == 'M') {
		v.x = maxboltz(opt.sqrtKT_m);
		v.y = maxboltz(opt.sqrtKT_m);
		v.z = maxboltz(opt.sqrtKT_m);
	}
	else if (coll_type == 'G' && opt.dist == 'C') {
		double3 vec;
		vec.x = normal(0.0, 1.0);
		vec.y = normal(0.0, 1.0);
		vec.z = normal(0.0, 1.0);
		double vec_norm = len(vec);
		v = Vel * vec/vec_norm;
	}
}

/*
Moves the particle based on the particle velocity and calcuated timestep.
*/
__PREPROCD__ void particle::move(options opt) {
	t_old = t; // update the time
	t += dt; //increment forward
	pos_old = pos; //update old position
	v_old = v; //update old velocity
	double3 a;
	if(opt.gravity)
		a = (double3){0.0, G_CONST, 0.0}; //acceleration due to gravity
	else
		a = {0.0, 0.0, 0.0};
	pos = pos_old +  v * dt + 0.5 *a*dt*dt; //update position
	v = v + a * dt; //update velocity
	
}

/*
Performs one particle and spin integration step.
*/

__PREPROCD__ void particle::step(options opt) {
	calc_next_collision_time(opt); //when do we hit something next?
	move(opt);
	new_velocities(opt);
	int spinResult = 0;
    double tempH = h;
	if(opt.integratorType == 0){
		//use the DOP853 algorithm for spin tracking
		spinResult = integrateDOP(t_old, t, S, pos_old, pos, v_old, v, opt, tempH);
	}
	else if(opt.integratorType == 1){
		//use the hybrid RK45 method
		spinResult = integrateRK45Hybrid(t_old, t, S, pos_old, pos, v_old, v, opt, tempH);
	}
	else if(opt.integratorType == 2){
		spinResult = integrateMagnusCFET(t_old, t, S, pos_old, pos, v_old, v, opt, tempH);
	}
	else{
		//this is an unrecognized option so just don't integrate the spin in this case
	}
    if(opt.keepStepSize)
        h = tempH;
    if (spinResult < 0){
        printf("%d %d\n", partID, spinResult);
        stopParticle = true;
        t = nan("");
        pos = (double3){nan(""), nan(""), nan("")};
        v = (double3){nan(""), nan(""), nan("")};
        S = (double3){nan(""), nan(""), nan("")};
    }
	n_steps += 1;
}

/*
Convenience function that calls the step() function repeatedL.y until the end time is reached.
*/

__PREPROCD__ outputDtype particle::getState(){
	outputDtype out;
	out.t = t;
	out.x = pos;
	out.v = v;
	out.s = S;
	return out;
}

__PREPROCD__ void particle::updateTF(double tfNew){
	dt = tfNew - tf;
	tf = tfNew;
	finished = false;
	return;
}

__PREPROCD__ void particle::run(options opt){
	finished = false;
	while (finished == false && stopParticle == false){
		step(opt);
	}
}

