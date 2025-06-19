#include "../include/rng.h"

__PREPROC__ uint64_t rol64(const uint64_t x, const int k) {
	return (x << k) | (x >> (64 - k));
}

__PREPROC__ uint64_t splitmix64(uint64_t& state) {
	uint64_t result = (state += 0x9E3779B97f4A7C15);
	result = (result ^ (result >> 30)) * 0xBF58476D1CE4E5B9;
	result = (result ^ (result >> 27)) * 0x94D049BB133111EB;
	return result ^ (result >> 31);
}

__PREPROC__ void initialize_xoshiro_state(rngState& state, uint64_t seed) {
	state.x = splitmix64(seed);
	state.y = splitmix64(seed);
	state.z = splitmix64(seed);
	state.w = splitmix64(seed);
	xoshiro256p(state);
	xoshiro256p(state);
	xoshiro256p(state);
}

__PREPROC__ uint64_t xoshiro256p(rngState& state) {
	//using this https://en.wikipedia.org/wiki/Xorshift#xoshiro256+
	const uint64_t result = state.x + state.w;
	const uint64_t t = state.y << 17;
	state.z ^= state.x;
	state.w ^= state.y;
	state.y ^= state.z;
	state.x ^= state.w;
	state.z ^= t;
	state.w = rol64(state.w, 45);
	return result;
}

__PREPROC__ _PREC uniform(rngState& state) {
	//A random number between low and high based on the xoshiro256** algorithm
	//https://en.wikipedia.org/wiki/Xorshift#xoshiro256**
	//https://prng.di.unimi.it/
	uint64_t temp = xoshiro256p(state);
	const double out = DoubleFromBits(temp);
	return out;
}

__PREPROC__ _PREC uniform(rngState& state, const _PREC low, const _PREC high) {
	return uniform(state)*(high-low)+low;
}

__PREPROC__ _PREC normal(rngState& state, const _PREC mean, const _PREC std) {
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

__PREPROC__ _PREC maxboltz(rngState& state, const _PREC sqrtkT_m) {
	return normal(state, 0.0, 1.0) * sqrtkT_m;
}

__PREPROC__ _PREC unif02pi(rngState& state) {
	return uniform(state) * 2.0 * M_PI;
}

__PREPROC__ _PREC exponential(rngState& state, const _PREC tc) {
	return - tc * log(1.0 - uniform(state));
}
