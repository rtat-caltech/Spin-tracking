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
	__PREPROCD__ _PREC sgn(const T val) {
	if(val<0)
		return -1.0;
	else
		return 1.0;
	//return (T(0) < val) - (val < T(0)); //we don't want the 0 case
}

__PREPROCD__ uint64_t rol64(const uint64_t x, const int k) {
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

__PREPROCD__ uint64_t xoshiro256p(rngState& state) {
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

__PREPROCD__ static inline double DoubleFromBits(const uint64_t i) {
	return (i >> 11) * 0x1.0p-53;
}

__PREPROCD__ _PREC uniform(rngState& state) {
	//A random number between low and high based on the xoshiro256** algorithm
	//https://en.wikipedia.org/wiki/Xorshift#xoshiro256**
	//https://prng.di.unimi.it/
	uint64_t temp = xoshiro256p(state);
	const double out = DoubleFromBits(temp);
	return out;
}

__PREPROCD__ _PREC uniform(rngState& state, const _PREC low, const _PREC high) {
	return uniform(state)*(high-low)+low;
}

__PREPROCD__ _PREC normal(rngState& state, const _PREC mean, const _PREC std) {
	//Generates a random value from a normal distribution using the Marsaglia Polar Method.
	//https://en.wikipedia.org/wiki/Marsaglia_polar_method
	if(state.hasSpare) {
		state.hasSpare = false;
		return state.spare * std + mean;
	} else {
		_PREC ts, tu, tv;
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

__PREPROCD__ _PREC maxboltz(rngState& state, const _PREC sqrtkT_m) {
	return normal(state, 0.0, 1.0) * sqrtkT_m;
}

__PREPROCD__ _PREC unif02pi(rngState& state) {
	return uniform(state) * 2.0 * M_PI;
}

__PREPROCD__ _PREC exponential(rngState& state, const _PREC tc) {
	return - tc * log(1.0 - uniform(state));
}

__PREPROCD__ void calc_next_collision_time(_PREC t, _PREC tf, coords v, coords pos, _PREC& next_gas_coll_time, _PREC& dt, char& coll_type, size_t &n_bounce, size_t &n_coll, bool& finished, char& wall_hit, rngState& state, const options opt) {
	_PREC dx, dy, dz, dtx, dty, dtz = (_PREC)0.0;
	if(opt.gravity) {
		//calculate distance to collision point
		dx = sgn(v.x) * opt.L.x / (_PREC)2.0 - pos.x;
		dz = sgn(v.z) * opt.L.z / (_PREC)2.0 - pos.z;

		//time to wall for x and z coordinate
		dtx = dx / v.x;
		dtz = dz / v.z;
		_PREC y2 = v.y * v.y;
		if(sgn(v.y) <= (_PREC)0.0) { //if the particle has negative y velocity
			dy = pos.y + opt.L.y*0.5; //distance to the bottom of the cell
			_PREC sqr = sqrt(-2.0*G_CONST*dy+y2);
			_PREC temp1 = -(sqr+v.y)/G_CONST;
			_PREC temp2 = (sqr-v.y)/G_CONST;
			dty = min(std::abs(temp1), std::abs(temp2));
		} else {
			_PREC maxHeight = -0.5 * y2/G_CONST + pos.y;
			if(maxHeight < 0.5 * opt.L.y) { //in this case it can't hit the ceiling, calculate time to the floor
				_PREC t1 = -v.y/G_CONST; //time until it stops moving upwards
				_PREC topOfFlight = pos.y+v.y*t1+0.5*G_CONST*t1*t1;//highest location in path
				_PREC t2 = sqrt((-0.5*opt.L.y - topOfFlight)*2.0/G_CONST); //time to fall to bottom of cell
				dty = t1 + t2; //total time for this path
			} else {
				_PREC sqr = sqrt(y2 - 2.0*G_CONST*(pos.y-(0.5*opt.L.y)));//specifically targetting hitting the ceiling
				_PREC temp1 = (-v.y + sqr)/G_CONST;
				_PREC temp2 = (-v.y-sqr)/G_CONST;
				dty = min(std::abs(temp1), std::abs(temp2));
			}
		}
		if (dtx < 1e-16 || std::isnan(dtx))
			dtx = 1e6;
		else if (dty < 1e-16 || std::isnan(dty))
			dty = 1e6;
		else if (dtz < 1e-16 || std::isnan(dtz))
			dtz = 1e6;
	} else {

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
	_PREC tbounce;
	if(dtx < dty && dtx < dtz) { //if the time to hit the x wall is the smallest, do this
		tbounce = dtx;
		wall_hit = 'x';
	} else if(dty < dtx && dty < dtz) { //if the time to hit the y wall is the smallest, do this
		tbounce = dty;
		wall_hit = 'y';
	} else if(dtz < dtx && dtz < dty) { //if the time to hit the z wall is the smallest, do this
		tbounce = dtz;
		wall_hit = 'z';
	} else {
		//if we reach this, then we have hit a corner
		if (dtx == dty && dtx == dtz) {
			tbounce = dtx;
			wall_hit = 'a'; //a means we hit a corner
		} else {
			if (dtx == dty) {
				tbounce = dtx;
				wall_hit = 'b'; //b means we hit the x-y edge
			} else if(dtx == dtz) {
				tbounce = dtx;
				wall_hit = 'c'; //c means we hit the x-z edge
			} else if(dty == dtz) {
				tbounce = dty;
				wall_hit = 'd'; //c means we hit the x-z edge
			}
		}
	}
	_PREC timeToNextGas = next_gas_coll_time - t; //calculate how long until the next gas collision
	int nextCol = -1;
	_PREC timeToNextCol = 0;
	//check if it were to hit something, which it would hit first
	if (timeToNextGas < tbounce) { //is a gas collision sooner than a wall collision?
		nextCol = 0; //gas collision next
		timeToNextCol = timeToNextGas;
	} else if(timeToNextGas == tbounce) { //in the case they were to happen at the same time, just do the wall collision
		//but also re-calculate the next time to the gas collision
		next_gas_coll_time += exponential(state, opt.tc);
		timeToNextCol = timeToNextGas;
		nextCol = 1;
	} else { //in this case the first collision is a wall bounce
		nextCol = 1;
		timeToNextCol = tbounce;
	}
	int nextNothing = -1;
	//if it were to not hit something, do we reach the end of the simulation first or the max step size?
	_PREC nextNoCol = 0;
	if (tf - t <= opt.maxPosStep) { //if the amount of time left is the same or less than the max step
		nextNothing = 0;
		nextNoCol = tf-t;
	} else { //otherwise the next step is capped by the max step size
		nextNothing = 1;
		nextNoCol = opt.maxPosStep;
	}
	//okay now check and see if we hit something before we want to output stuff
	if(timeToNextCol <= nextNoCol) {
		//if the shortest time is the collision, or if it's a tie, assume the collision case and do the updates
		//in this case we specify what type of collision it was and do the normal collision behavior
		dt = timeToNextCol;
		if(nextCol == 0) { //this is a gas/phonon collision
			coll_type = 'G';
			n_coll += 1;
			next_gas_coll_time += exponential(state, opt.tc);
		} else {
			n_bounce += 1;
			coll_type = 'W';
		}
		if (timeToNextCol == nextNoCol) { //check if they were equal so we can set the output states properly
			if(nextNothing == 0) { //if we reached the end of the simulation
				finished = true;
			}
			//in the case of the max position step size, just don't do anything and let it keep on going
		}
	} else {
		//in this case we didn't have a collision and reached either the max step size or the end of the simulation
		coll_type = 'N';
		dt = nextNoCol;
		if(nextNothing == 0) {
			finished = true;
		}
	}

}


/*
  Update the position and velocity after the collision that is found
*/
__PREPROCD__ void update_position_and_velocity(_PREC &t_old, _PREC &t, _PREC &dt, coords &pos_old, coords &pos, coords &v_old, coords &v, char& coll_type, char& wall_hit, rngState& state, bool &stopParticle, const options opt) {
	v_old = v;

	t_old = t; // update the time
	t += dt; //increment forward
	pos_old = pos; //update old position
	v_old = v; //update old velocity
	coords a;
	if(opt.gravity)
		a = (coords) {
			0.0, G_CONST, 0.0
		}; //acceleration due to gravity
	else
		a = (coords) {
			0.0, 0.0, 0.0
		};
	pos = pos_old +  v * dt + 0.5 * a * dt * dt; //update position
	v = v + a * dt; //update velocity
	_PREC Vel = len(v);
	if (coll_type == 'N') {
		//in this case we don't have a wall collision and it's just iterating through space still
		//don't mess with the position or velocity, should be fine
	} else if(coll_type == 'W') { //wall collision detected
		if (wall_hit == 'x') {
			pos.x = sgn(pos.x)*opt.L.x*0.5;
		} else if (wall_hit == 'y') {
			pos.y = sgn(pos.y)*opt.L.y*0.5;
		} else if (wall_hit == 'z') {
			pos.z = sgn(pos.z)*opt.L.z*0.5;
		} else if (wall_hit == 'a') {
			pos.x = sgn(pos.x)*opt.L.x*0.5;
			pos.y = sgn(pos.y)*opt.L.y*0.5;
			pos.z = sgn(pos.z)*opt.L.z*0.5;
		} else if (wall_hit == 'b') {
			pos.x = sgn(pos.x)*opt.L.x*0.5;
			pos.y = sgn(pos.y)*opt.L.y*0.5;
		} else if (wall_hit == 'c') {
			pos.x = sgn(pos.x)*opt.L.x*0.5;
			pos.z = sgn(pos.z)*opt.L.z*0.5;
		} else if (wall_hit == 'd') {
			pos.y = sgn(pos.y)*opt.L.y*0.5;
			pos.z = sgn(pos.z)*opt.L.z*0.5;
		}
		if(opt.diffuse > FLT_MIN) {
			//could maybe have diffuse scattering, so sample the RNG to see if it happens
			bool diffuse = (_PREC)uniform(state) < (_PREC)opt.diffuse;
			if(diffuse) { //if we want to do a diffuse collision, do this
				//V = sqrt(vx * vx + vy * vy + vz * vz);
				_PREC phi,theta;
				phi = acos(sqrt(uniform(state)));
				theta = unif02pi(state);
				if (wall_hit == 'x') {
					v.x = -1 * sgn(v.x) * Vel * cos(phi);
					v.y = -Vel * sin(phi) * cos(theta);
					v.z = Vel * sin(phi) * sin(theta);
				} else if (wall_hit == 'y') {
					v.x = Vel * sin(phi) * cos(theta);
					v.y = -1 * sgn(v.y) * Vel * cos(phi);
					v.z = Vel * sin(phi) * sin(theta);
				} else if (wall_hit == 'z') {
					v.x = Vel * sin(phi) * cos(theta);
					v.y = Vel * sin(phi) * sin(theta);
					v.z = -1 * sgn(v.z) * Vel * cos(phi);
				} else {
					stopParticle = true;
				}
			} else { //otehrwise just flip the velocities around and call it a day
				if (wall_hit == 'x')
					v.x *= -1.0;
				else if (wall_hit == 'y')
					v.y *= -1.0;
				else if (wall_hit == 'z')
					v.z *= -1.0;
				else if (wall_hit == 'a') {
					v.x *= -1.0;
					v.y *= -1.0;
					v.z *= -1.0;
				} else if (wall_hit == 'b') {
					v.x *= -1.0;
					v.y *= -1.0;
				} else if (wall_hit == 'c') {
					v.x *= -1.0;
					v.z *= -1.0;
				} else if (wall_hit == 'd') {
					v.y *= -1.0;
					v.z *= -1.0;
				}
			}
		} else { //otehrwise just flip the velocities around and call it a day
			if (wall_hit == 'x')
				v.x *= -1.0;
			else if (wall_hit == 'y')
				v.y *= -1.0;
			else if (wall_hit == 'z')
				v.z *= -1.0;
			else if (wall_hit == 'a') {
				v.x *= -1.0;
				v.y *= -1.0;
				v.z *= -1.0;
			} else if (wall_hit == 'b') {
				v.x *= -1.0;
				v.y *= -1.0;
			} else if (wall_hit == 'c') {
				v.x *= -1.0;
				v.z *= -1.0;
			} else if (wall_hit == 'd') {
				v.y *= -1.0;
				v.z *= -1.0;
			}
		}
	} else if(coll_type == 'G') {
		//in this case it's a "gas"/phonon collision
		if (opt.dist == 'M') {
			v.x = maxboltz(state, opt.sqrtKT_m);
			v.y = maxboltz(state, opt.sqrtKT_m);
			v.z = maxboltz(state, opt.sqrtKT_m);
		} else if(opt.dist == 'C') {
			coords vec;
			vec.x = normal(state, 0.0, 1.0);
			vec.y = normal(state, 0.0, 1.0);
			vec.z = normal(state, 0.0, 1.0);
			_PREC vec_norm = len(vec);
			v = Vel * vec/vec_norm;
		}
	}
}

/*
  Moves the particle based on the particle velocity and calcuated timestep.
*/
__PREPROCD__ void move(_PREC &t_old, _PREC& t, coords &pos_old, coords &pos, coords& v, coords& v_old, _PREC &dt, const options opt) {
	t_old = t; // update the time
	t += dt; //increment forward
	pos_old = pos; //update old position
	v_old = v; //update old velocity
	coords a;
	if(opt.gravity)
		a = (coords) {
			0.0, G_CONST, 0.0
		}; //acceleration due to gravity
	else
		a = (coords) {
			0.0, 0.0, 0.0
		};
	pos = pos_old +  v * dt + 0.5 * a * dt * dt; //update position
	v = v + a * dt; //update velocity

}

__PREPROCD__ void sanity_check(_PREC &t_old, _PREC& t, coords &pos_old, coords &pos, coords& v, coords& v_old, char& coll_type, char& wall_hit, bool &stopParticle, int &failureState, const options opt) {
	_PREC tol = 1.0e-15;
	if(abs(pos.x) > opt.L.x/2.0+tol) {
		stopParticle = true;
		failureState = 1; //out of bounds position found
		return;
	}
	if(abs(pos.y) > opt.L.y/2.0+tol) {
		stopParticle = true;
		failureState = 1; //out of bounds position found
		return;
	}
	if(abs(pos.z) > opt.L.z/2.0+tol) {
		stopParticle = true;
		failureState = 1; //out of bounds position found
		return;
	}
	if(isinf(t_old) || isinf(t)) {
		stopParticle = true;
		failureState = 2; //infinite time found
		return;
	}
	if(isnan(pos_old.x) || isnan(pos_old.y) || isnan(pos_old.z) || isnan(pos.x) || isnan(pos.y) || isnan(pos.z)) {
		stopParticle = true;
		failureState = 3; //nan position found
		return;
	}
	if(isnan(v.x) || isnan(v.y) || isnan(v.z) || isnan(v_old.x) || isnan(v_old.y) || isnan(v_old.z)) {
		failureState = 4; //nan velocity found
		stopParticle = true;
		return;
	}
}
/*
  Convenience function that calls the step() function repeatedL.y until the end time is reached.
*/

#if defined(__HIPCC__) || defined(__NVCOMPILER) || defined(__NVCC__)
__global__ void initParticlesGPU(options opt, coords *S, coords *v, coords *v_old,
                                 coords *pos, coords *pos_old, _PREC *t, _PREC *t_old,
                                 _PREC *tf, _PREC *dt, _PREC *next_gas_coll_time, _PREC *h,
                                 rngState *state, size_t *n_bounce, size_t *n_coll, size_t *n_steps,
                                 unsigned int *partID, int * failureState, bool *stopParticle, char *coll_type, char *wall_hit, SpectrumAggregator *specagg) {
	unsigned int ipart = threadIdx.x + blockIdx.x * blockDim.x;
	if(ipart < opt.numParticles) {
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
		coords tempos;
		tempos.x = uniform(rng)*opt.L.x-opt.L.x/2.0;
		tempos.y = uniform(rng)*opt.L.y-opt.L.y/2.0;
		tempos.z = uniform(rng)*opt.L.z-opt.L.z/2.0;

		pos[ipart] = tempos;
		pos_old[ipart] = tempos;

		coords tempv;
		if (opt.gas_coll == true) {
			next_gas_coll_time[ipart] = exponential(rng, opt.tc);
		} else
			next_gas_coll_time[ipart] = DBL_MAX; //basically set to the end of time
		if (opt.dist == 'C') {
			if(std::abs(opt.V) <= 1.0e-6) {
				//assume the user meant 0 speed at this point
				tempv = (coords) {
					0.0, 0.0, 0.0
				};
			} else {
				tempv.x = normal(rng, 0.0, 1.0);
				tempv.y = normal(rng, 0.0, 1.0);
				tempv.z = normal(rng, 0.0, 1.0);
				_PREC vec_norm = len(tempv);
				tempv = opt.V * tempv/vec_norm;
			}
		} else if (opt.dist == 'M') {
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
		specagg[ipart] = SpectrumAggregator();
	}
}

__global__ void runSimulationGPU(options opt, coords *pS, coords *pv, coords *pv_old,
                                 coords *ppos, coords *ppos_old, _PREC *pt, _PREC *pt_old,
                                 _PREC *ptf, _PREC *pdt, _PREC *pnext_gas_coll_time, _PREC *ph,
                                 rngState *pstate, size_t *pn_bounce, size_t *pn_coll, size_t *pn_steps,
                                 unsigned int *ppartID, int* pfailureState, bool *pstopParticle, char *pcoll_type, char *pwall_hit, SpectrumAggregator *pspecagg,
                                 _PREC nextTOut) {
	unsigned int ipart = threadIdx.x + blockIdx.x * blockDim.x;
	if(ipart < opt.numParticles) {
		//load the individual particle data
		coords S = pS[ipart];
		coords v = pv[ipart];
		coords v_old = pv_old[ipart];
		coords pos = ppos[ipart];
		coords pos_old = ppos_old[ipart];

		_PREC t = pt[ipart];
		_PREC t_old = pt_old[ipart];
		_PREC tf; //= ptf[ipart]; // no need to load it's done below
		_PREC dt; //= pdt[ipart]; //no need to load it's defined below
		_PREC next_gas_coll_time = pnext_gas_coll_time[ipart];
		_PREC h = ph[ipart];

		rngState state = pstate[ipart];
		size_t n_bounce = pn_bounce[ipart];
		size_t n_coll = pn_coll[ipart];
		size_t n_steps = pn_steps[ipart];
		unsigned int partID = ppartID[ipart];
		int failureState = pfailureState[ipart];
		bool stopParticle = pstopParticle[ipart];
		char coll_type = pcoll_type[ipart];
		char wall_hit = pwall_hit[ipart];
		SpectrumAggregator specagg = pspecagg[ipart];
		specagg.reset();
		dt = nextTOut - t;
		tf = nextTOut;
		bool finished = false;
		int spinResult = 0;
		_PREC tempH = h;
		//now start the actual integration and tracking process
		while(finished == false && stopParticle == false) {
			calc_next_collision_time(t, tf, v, pos, next_gas_coll_time, dt, coll_type, n_bounce, n_coll, finished, wall_hit, state, opt);
			update_position_and_velocity(t_old, t, dt, pos_old, pos, v_old, v, coll_type, wall_hit, state, stopParticle, opt);
			sanity_check(t_old, t, pos_old, pos, v, v_old, coll_type, wall_hit, stopParticle, failureState, opt);
			if(failureState == 0) {
				if(opt.integratorType == 0) {
					//use the DOP853 algorithm for spin tracking
					spinResult = integrateDOP(t_old, t, S, pos_old, pos, v_old, v, opt, tempH);
				} else if(opt.integratorType == 1) {
					//use the default RK45 method, no quaternions or anything
					spinResult = integrateRK45(t_old, t, S, pos_old, pos, v_old, v, opt, tempH);
				} else if(opt.integratorType == 2) {
					//use the MagnusCFET method
					spinResult = integrateMagnusCFET(t_old, t, S, pos_old, pos, v_old, v, opt, tempH);
				} else if(opt.integratorType == 3) {
					//rk45 method but with quaternions instead
					spinResult = integrateRK45Quaternion(t_old, t, S, pos_old, pos, v_old, v, opt, tempH);
				} else if(opt.integratorType == 4) {
					//different set of coefficients for RK45
					spinResult = integrateRKF45(t_old, t, S, pos_old, pos, v_old, v, opt, tempH);
				} else if(opt.integratorType == 5) {
					//same as option 3 but for option 4's coefficients
					spinResult = integrateRKF45Quaternion(t_old, t, S, pos_old, pos, v_old, v, opt, tempH);
				} else if(opt.integratorType == 6) {
					spinResult = integrateSpectrum(t_old, t, specagg, pos_old, pos, v_old, v, opt, tempH);
				} else {
					//do nothing
					spinResult = 0;
				}
				if(opt.keepStepSize) {
					h = tempH;
				}
				if (spinResult < 0) {
					stopParticle = true;
					failureState = spinResult;
				}
			}
			if(stopParticle) {
				t = nan("");
				pos = (coords) {
					nan(""), nan(""), nan("")
				};
				v = (coords) {
					nan(""), nan(""), nan("")
				};
				S = (coords) {
					nan(""), nan(""), nan("")
				};
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
		pfailureState[ipart] = failureState;
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
void initParticlesCPU(options opt, coords *pS, coords *pv, coords *pv_old,
                      coords *ppos, coords *ppos_old, _PREC *pt, _PREC *pt_old,
                      _PREC *ptf, _PREC *pdt, _PREC *pnext_gas_coll_time, _PREC *ph,
                      rngState *pstate, size_t *pn_bounce, size_t *pn_coll, size_t *pn_steps,
                      unsigned int *ppartID, int *pfailureState, bool *pstopParticle, char *pcoll_type, char *pwall_hit, SpectrumAggregator *specagg) {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
	for(int ipart = 0; ipart < opt.numParticles; ipart++) {
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
		coords tempos;
		tempos.x = uniform(rng)*opt.L.x-opt.L.x/2.0;
		tempos.y = uniform(rng)*opt.L.y-opt.L.y/2.0;
		tempos.z = uniform(rng)*opt.L.z-opt.L.z/2.0;
		ppos[ipart] = tempos;
		ppos_old[ipart] = tempos;
		if (opt.gas_coll == true) {
			pnext_gas_coll_time[ipart] = exponential(rng, opt.tc);
		} else
			pnext_gas_coll_time[ipart] = DBL_MAX; //basically set to the end of time
		if (opt.dist == 'C') {
			coords vec;
			vec.x = normal(rng, 0.0, 1.0);
			vec.y = normal(rng, 0.0, 1.0);
			vec.z = normal(rng, 0.0, 1.0);

			_PREC vec_norm = len(vec);
			pv[ipart] = opt.V * vec/vec_norm;
		} else if (opt.dist == 'M') {
			coords vec;
			vec.x = maxboltz(rng, opt.sqrtKT_m);
			vec.y = maxboltz(rng, opt.sqrtKT_m);
			vec.z = maxboltz(rng, opt.sqrtKT_m);
			pv[ipart] = vec;
		}
		pv_old[ipart] = pv[ipart];
		pstate[ipart] = rng; //save the final rng state after all this stuff
		//initialize the rest of the particle data to standard values
		pS[ipart] = opt.yi;
		pt[ipart] = opt.t0;
		pt_old[ipart] = opt.t0;
		ptf[ipart] = opt.tf;
		pdt[ipart] = opt.h;
		ph[ipart] = opt.h;
		pstopParticle[ipart] = false;
		pcoll_type[ipart] = 'W';
		pfailureState[ipart] = 0;
		specagg[ipart] = SpectrumAggregator();
	}
}

void runSimulationCPU(options opt, coords *pS, coords *pv, coords *pv_old,
                      coords *ppos, coords *ppos_old, _PREC *pt, _PREC *pt_old,
                      _PREC *ptf, _PREC *pdt, _PREC *pnext_gas_coll_time, _PREC *ph,
                      rngState *pstate, size_t *pn_bounce, size_t *pn_coll, size_t *pn_steps,
                      unsigned int *ppartID, int *pfailureState, bool *pstopParticle, char *pcoll_type, 
                      char *pwall_hit, SpectrumAggregator *pspecagg, CovarianceSpectrum& cspec, _PREC nextTOut) {
	CovarianceSpectrum* cspec_array = (CovarianceSpectrum*) malloc(sizeof(CovarianceSpectrum) * opt.numParticles);	  
#if defined(_OPENMP)
#pragma omp parallel for
#endif
	for(unsigned int ipart = 0; ipart < opt.numParticles; ipart++) {
		//load the individual particle data
		coords S = pS[ipart];
		coords v = pv[ipart];
		coords v_old = pv_old[ipart];
		coords pos = ppos[ipart];
		coords pos_old = ppos_old[ipart];

		_PREC t = pt[ipart];
		_PREC t_old = pt_old[ipart];
		_PREC tf; //= ptf[ipart]; // no need to load it's done below
		_PREC dt; //= pdt[ipart]; //no need to load it's defined below
		_PREC next_gas_coll_time = pnext_gas_coll_time[ipart];
		_PREC h = ph[ipart];

		rngState state = pstate[ipart];
		size_t n_bounce = pn_bounce[ipart];
		size_t n_coll = pn_coll[ipart];
		size_t n_steps = pn_steps[ipart];
		unsigned int partID = ppartID[ipart];
		int failureState = pfailureState[ipart];
		bool stopParticle = pstopParticle[ipart];
		char coll_type = pcoll_type[ipart];
		char wall_hit = pwall_hit[ipart];
		SpectrumAggregator specagg = pspecagg[ipart];
		specagg.reset();
		dt = nextTOut - t;
		tf = nextTOut;
		bool finished = false;
		int spinResult = 0;
		_PREC tempH = h;
		//now start the actual integration and tracking process
		while(finished == false && stopParticle == false) {
			calc_next_collision_time(t, tf, v, pos, next_gas_coll_time, dt, coll_type, n_bounce, n_coll, finished, wall_hit, state, opt);
			update_position_and_velocity(t_old, t, dt, pos_old, pos, v_old, v, coll_type, wall_hit, state, stopParticle, opt);
			sanity_check(t_old, t, pos_old, pos, v, v_old, coll_type, wall_hit, stopParticle, failureState, opt);
			if(failureState == 0) {
				if(opt.integratorType == 0) {
					//use the DOP853 algorithm for spin tracking
					spinResult = integrateDOP(t_old, t, S, pos_old, pos, v_old, v, opt, tempH);
				} else if(opt.integratorType == 1) {
					//use the default RK45 method, no quaternions or anything
					spinResult = integrateRK45(t_old, t, S, pos_old, pos, v_old, v, opt, tempH);
				} else if(opt.integratorType == 2) {
					//use the MagnusCFET method
					spinResult = integrateMagnusCFET(t_old, t, S, pos_old, pos, v_old, v, opt, tempH);
				} else if(opt.integratorType == 3) {
					//rk45 method but with quaternions instead
					spinResult = integrateRK45Quaternion(t_old, t, S, pos_old, pos, v_old, v, opt, tempH);
				} else if(opt.integratorType == 4) {
					//different set of coefficients for RK45
					spinResult = integrateRKF45(t_old, t, S, pos_old, pos, v_old, v, opt, tempH);
				} else if(opt.integratorType == 5) {
					//same as option 3 but for option 4's coefficients
					spinResult = integrateRKF45Quaternion(t_old, t, S, pos_old, pos, v_old, v, opt, tempH);
				} else if(opt.integratorType == 6) {
					integrateSpectrum(t_old, t, specagg, pos_old, pos, v_old, v, opt, tempH);
				} else {
					//do nothing
					spinResult = 0;
				}
				if(opt.keepStepSize)
					h = tempH;
				if (spinResult < 0) {
					stopParticle = true;
					failureState = spinResult;
				}
			}
			if(stopParticle) {
				t = nan("");
				pos = (coords) {
					nan(""), nan(""), nan("")
				};
				v = (coords) {
					nan(""), nan(""), nan("")
				};
				S = (coords) {
					nan(""), nan(""), nan("")
				};
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
		pfailureState[ipart] = failureState;
		pn_bounce[ipart] = n_bounce;
		pn_coll[ipart] = n_coll;
		pn_steps[ipart] = n_steps;
		ppartID[ipart] = partID;
		pstopParticle[ipart] = stopParticle;
		pcoll_type[ipart] = coll_type;
		pwall_hit[ipart] = wall_hit;
		ptf[ipart] = tf;
		pspecagg[ipart] = specagg;
		cspec_array[ipart] = specagg.get_covariance_spectrum();
	}
	for(unsigned int ipart = 0; ipart < opt.numParticles; ipart++) {
		for (unsigned int s = 1; s < opt.numParticles; s *= 2) {
			if (ipart % (2 * s) == 0) {
				cspec_array[ipart].add(cspec_array[ipart + s]);
			}
		}
	}
	cspec.add(cspec_array[0]);
	free(cspec_array);
}

coords particle::floquetResults() {
	cspec.normalize();
	double Delta[2][2][NK] = {{{0}}};
	double X[2][2][NK] = {{{0}}};
	double Gamma[2][2][NK] = {{{0}}};
	double A[2][2] = {{0}};
  
	vector<pair<quaternion, Spectrum>> specs = cspec.extract();
	Matrix2cd rho = bloch_to_density(opt.yi, fd.f_modes_0);
	for (int i = 0; i < specs.size(); i++) {
		quaternion c_op = specs.at(i).first;
		Spectrum spec = specs.at(i).second;
		floquet_master_equation_rates(fd, c_op, 2*M_PI/opt.w,spec, 
		                              Delta, X, Gamma, A);
	}
	rho = integrateFloquetMarkov(opt.t0, opt.tf, rho, A);
	int n_period = round((opt.tf - opt.t0) * opt.w/(2 * M_PI));
	return density_to_bloch(rho, fd.f_modes_0 * pow(fd.f_energies, n_period));
}

void particle::postProcess(FILE *f) {
	if (opt.integratorType == 6) {
		coords b_end = floquetResults();
		fwrite(&b_end, sizeof(coords), 1, f);
		cout << "Final Bloch Vector:" << endl;
		cout << b_end << endl;
	}
}
#endif

void particle::runSimulation(_PREC nextTOut){
#if defined(__HIPCC__) || defined(__NVCOMPILER) || defined(__NVCC__)
	runSimulationGPU<<<numBlocks, numPartsPerBlock>>>(opt, S, v, v_old, pos, pos_old, t, 
		t_old, tf, dt, next_gas_coll_time, h, state, n_bounce, n_coll, n_steps,
		partID, failureState, stopParticle, coll_type, wall_hit, specagg, nextTOut);
#else
	runSimulationCPU(opt, S, v, v_old, pos, pos_old, t, 
	                 t_old, tf, dt, next_gas_coll_time, h, state, n_bounce, n_coll, n_steps,
	                 partID, failureState, stopParticle, coll_type, wall_hit, specagg, cspec, nextTOut);
#endif
        
};


coords particle::spinMean() {
	coords S_sum = {0, 0, 0};
	for (int i = 0; i < opt.numParticles; i++) {
		S_sum = S_sum + S[i];
	}
	return S_sum/opt.numParticles;
}

floquetDiagonalization particle::initializeSpectra(CovarianceSpectrum& cspec, options OPT) {
	double t0 = 0.0;
	double tf = (2*M_PI)/OPT.w; //TODO
	int n_prop = 100;
	quaternion* propagators = (quaternion*) malloc(sizeof(quaternion) * n_prop);
	quaternion y = {1, 0, 0, 0};
	double h = 1e-6;
	for (int i=0; i < n_prop; i++) {
		double t1 = t0 + (tf - t0) * i/n_prop;
		double t2 = t0 + (tf - t0) * (i+1)/n_prop;
		integrateHamiltonian(t1, t2, y, OPT, h);
		propagators[i] = y;
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
	for(unsigned int tid = 0; tid < OPT.numParticles; tid++) {
		specagg[tid].initialize(frequencies, (tf - t0)/n_prop);
	}
	cspec.initialize(frequencies, (tf - t0)/n_prop);

	floquetDiagonalization fd;
	fd.propagators = propagators;
	fd.f_modes_0 = eigen_vectors;
	fd.f_energies = eigen_values;
	fd.n_prop = n_prop;
	return fd;
}


void particle::aggregateSpectrum(CovarianceSpectrum& cspec, int numParticles) {
	for(unsigned int tid = 0; tid < numParticles; tid++) {
		cspec.add(specagg[tid].get_covariance_spectrum());
	}
}

SpectrumAggregator* particle::getSpectrumAggregators() {
	return specagg;
}

coords* particle::getVelocities() {
	return v;
}
