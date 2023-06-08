'''
This file defines an implementation of the XORWOW random number generator.
This is done so both the CPU and GPU can use the same routine for random values.

It's defined in a way that is compatible with both GPU and CPU implementations through Numba
'''

import numba
from numba import cuda
import numpy as np

@cuda.jit(numba.uint64(numba.uint64))
def initialize_splitmix64(seed):
	result = seed + 0x9E3779B97f4A7C15
	result = (result ^ (result >> 30)) * 0xBF58476D1CE4E5B9;
	result = (result ^ (result >> 27)) * 0x94D049BB133111EB;
	return result ^ (result >> 31);

@numba.jit
def xorshift128_init(seed):
	#first 64-bits
	tmp1 = initialize_splitmix64(seed)
	#second 64bits
	seed+=0x9E3779B97f4A7C15
	tmp2 = initialize_splitmix64(seed);
	#third 64bits
	seed+=0x9E3779B97f4A7C15
	tmp3 = initialize_splitmix64(seed);
	#4th 64bits
	seed+=0x9E3779B97f4A7C15
	tmp4 = initialize_splitmix64(seed);
	return (tmp1, tmp2, tmp3, tmp4)

@numba.jit(numba.uint64(numba.uint64, numba.int32))
def rol64(x, k):
	return (x<<k) | (x>>(64-k))

@numba.jit(numba.uint64(numba.uint64[:]))
def xoshiro256ss(state):
	a, b, c, d = state
	result = rol64(b*5, 7)*9
	t = b << 17
	c ^= a;
	d ^= b;
	b ^= c;
	a ^= d;
	c ^= t;
	d = rol64(d, 45);
	state[:] = (a, b, c, d)
	return result

@numba.jit(numba.uint64(numba.uint64[:]))
def xoshiro256ss(state):
	a, b, c, d = state
	result = rol64(b*5, 7)*9
	t = b << 17
	c ^= a;
	d ^= b;
	b ^= c;
	a ^= d;
	c ^= t;
	d = rol64(d, 45);
	state[:] = (a, b, c, d)
	return result

@numba.jit(numba.float64(numba.uint64[:]))
def uniform(state):
	val = xoshiro256ss(state)
	return float(val/18446744073709551615)

@cuda.jit(numba.float64(numba.uint64[:]), device=True)
def uniform(state):
	val = xoshiro256ss(state)
	return float(val/18446744073709551615)