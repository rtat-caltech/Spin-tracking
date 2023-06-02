'''
This file defines an implementation of the XORWOW random number generator.
This is done so both the CPU and GPU can use the same routine for random values.

It's defined in a way that is compatible with both GPU and CPU implementations through Numba

This is based on the method described here: https://en.wikipedia.org/wiki/Xorshift#xoshiro256**
'''

import numba
import numpy as np

@numba.jit(locals={'result': numba.uint64, 'seed': numba.uint64})
def initialize_splitmix64(seed):
	result = seed + 0x9E3779B97f4A7C15
	result = (result ^ (result >> 30)) * 0xBF58476D1CE4E5B9;
	result = (result ^ (result >> 27)) * 0x94D049BB133111EB;
	return result ^ (result >> 31);

@numba.jit(locals={'seed': numba.uint64, 'tmp1': numba.uint64, 'tmp2': numba.uint64, 'tmp3': numba.uint64, 'tmp4': numba.uint64, 'out': numba.uint64[:]})
def xorshift128_init(seed, state):
	#first 64-bits
	tmp1 = np.uint64(initialize_splitmix64(seed))
	#second 64bits
	seed+=0x9E3779B97f4A7C15
	tmp2 = np.uint64(initialize_splitmix64(seed))
	#third 64bits
	seed+=0x9E3779B97f4A7C15
	tmp3 = np.uint64(initialize_splitmix64(seed))
	#4th 64bits
	seed+=0x9E3779B97f4A7C15
	tmp4 = np.uint64(initialize_splitmix64(seed))
	state[0] = tmp1
	state[1] = tmp2
	state[2] = tmp3
	state[3] = tmp4
	return

@numba.jit(locals={'x': numba.uint64, 'k': numba.uint64})
def rol64(x, k):
	return (np.uint64(x)<<np.uint64(k) | (np.uint64(x)>>np.uint64(64-k)))

@numba.jit(locals={'state': numba.uint64[:], 'a': numba.uint64, 'b': numba.uint64, 'c': numba.uint64, 'd': numba.uint64, 't': numba.uint64, 'result': numba.uint64,'def5': numba.uint64, 'def7': numba.uint64, 'def9': numba.uint64})
def xoshiro256ss(state):
	a, b, c, d = state
	def5 = 5 * b
	result = rol64(def5, 7)*9
	t = b << 17
	c ^= a;
	d ^= b;
	b ^= c;
	a ^= d;
	c ^= t;
	d = rol64(d, 45);
	state[0] = a
	state[1] = b
	state[2] = c
	state[3] = d
	return result

@numba.jit
def uniform(state, low = 0.0, high = 1.0):
	"""
	Generates a uniform random value between low and high
	
	Parameters
	----------
	state
		The random number generator state as created by the rng.xorshift128_init function
		Must be passed directly and not via state[:] or np.copy(state). Must be passed as state
		otherwise the pass by reference behavior is broken and the state won't automatically update.
	
	low: defaults to 0.0
		The smallest value of the output
	high: defaults to 1.0
		The largest value of the output
	
	Returns
	--------
	A random number between low and high based on the xoshiro256** algorithm
	https://en.wikipedia.org/wiki/Xorshift#xoshiro256**
	
	"""
	val = np.uint64(xoshiro256ss(state))
	t = val/np.uint64(18446744073709551615)*(high-low) + low
	return t

@numba.jit
def normal(state, mean=0.0, std=1.0):
	"""
	Generates a random value from a normal distribution using the Marsaglia Polar Method.
	https://en.wikipedia.org/wiki/Marsaglia_polar_method
	
	Parameters
	----------
	state:
		The random number generator state as created by the rng.xorshift128_init function
		Must be passed directly and not via state[:] or np.copy(state). Must be passed as state
		otherwise the pass by reference behavior is broken and the state won't automatically update.
	
	mean:
		The mean of the distribution. Defaults to 0.0
	
	std:
		The standard deviation of the distribution. Defaults to 1.0
	
	Returns
	-------
	A random number from the normal distribution.
	"""
	s = 2
	while(s >= 1.0 or s == 0.0):
		u = uniform(state, -1.0, 1.0)
		v = uniform(state, -1.0, 1.0)
		s = u*u + v*v
	s = np.sqrt(-2.0*np.log(s)/s)
	return mean + std * u * s

@numba.jit
def maxboltz(state, sqrtkT_m):
	return normal(state) * sqrtkT_m

@numba.jit
def exponential(state, tc):
	return -tc * np.log(1.0 - uniform(state))