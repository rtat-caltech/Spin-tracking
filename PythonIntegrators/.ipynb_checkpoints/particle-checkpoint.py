'''
This file defines the particle data type as well as a bunch of functions to handle the position tracking.

All of these should be defined so they are interchangeable between CPU and GPU if possible. 
'''

import numpy as np
import numba
import rng
import spinIntegrators as spin

particleDtype = np.dtype([
	('t', np.float64), #what time is it
	('x', np.float64, 3), #where is the particle now
	('v', np.float64, 3), #it's velocity
	('s', np.float64, 3), #it's spin
	('mass', np.float64), #it's mass
	('rng', np.uint64, 4), #rng seed
	('x_old', np.float64, 3), #previous position
	('v_old', np.float64, 3), #previous velocity
	('temperature', np.float64), #temperature
	('gamma', np.float64), #gyromagnetic ratio
	('tc', np.float64), #collision time constant
	('next_gas_coll_time', np.float64), #next time for a gas collision
	('coll_type', np.int8), #collision type, 0 is no collision, 1 is wall, 2 is gas
	('wall_hit', np.int8), #which wall it hit
	('finished', np.int8), #is the simulation for this particle done
	('n_bounce', np.int64), #number of wall bounces
	('n_coll', np.int64), #number of collisions
])

@numba.jit
def createParticle(mass=2.2*5e-27, t0=0.0, temperature=1.0, velocity_dist = 'fixed', vel=5.0, 
				   gamma = -2.078e8, cell=(0.07, 0.1, 0.4), position_dist='uniform', 
				   pos = (0, 0, 0), spin=(1.0, 0.0, 0.0), seed=0):
	"""
	This function initializes a particle
	
	Parameters
	----------
	mass
		The mass of the particle in kg
	t0
		The time the particle starts at in seconds
	temperature
		The temperature of the particle in Kelvin
	velocity_dist
		The method to generate the velocity of the particle
		If 'fixed', the vel parameter is used to fix the velocity, but direction is random
		If 'maxwell', samples a maxwell boltzman distribution based on mass and temperature to generate a random velocity
		If 'custom', the vel parameter sets the velocity
	vel
		a float that determines the magnitude of the velocity
		If the velocity_dist parameter is 'custom', this must be a 3 element tuple
	gamma
		Gyromagnetic ratio of the particle in rad/s/Tesla
		Defaults to -2.078e8
	cell
		The size of the simulation cell in meters as a 3 element tuple
	position_dist
		Determines how the start positions are determined
		Defaults to 'uniform', only supported option for now
	spin
		Sets the initial spin
		Defaults to (1.0, 0.0, 0.0)
	seed
		The random number generator seed for this particle
		This should be unique for all particles
	
	Returns
	-------
	Particle
		A numpy array containing the particle
	"""
	k = 1.380649e-23
	particle = np.zeros(1, dtype=particleDtype)[0]
	rng.xorshift128_init(seed, particle['rng']) #initialize the random numbers
	particle['t'] = t0 #set the start time
	#determine the starting position
	if position_dist == 'uniform':
		tx = rng.uniform(particle['rng'], -cell[0]/2.0, cell[0]/2.0)
		ty = rng.uniform(particle['rng'], -cell[1]/2.0, cell[1]/2.0)
		tz = rng.uniform(particle['rng'], -cell[2]/2.0, cell[2]/2.0)
		particle['x'][:] = (tx, ty, tz)
	elif position_dist == 'custom':
		particle['x'][:] = pos
	else:
		particle['x'][:] = (0.0, 0.0, 0.0)
	#determine the starting velocity
	if velocity_dist == 'fixed':
		tx = rng.uniform(particle['rng'])
		ty = rng.uniform(particle['rng'])
		tz = rng.uniform(particle['rng'])
		tv = np.sqrt(tx**2.0 + ty**2.0 + tz**2.0)
		scale = vel/tv
		particle['v'][:] = (scale*tx, scale*ty, scale*tz)
	elif velocity_dist == 'maxwell':
		sqrtKT_m = np.sqrt(k*temperature/mass)
		tx = rng.normal(particle['rng'])*sqrtKT_m
		ty = rng.normal(particle['rng'])*sqrtKT_m
		tz = rng.normal(particle['rng'])*sqrtKT_m
		particle['v'][:] = (tx, ty, tz)
	elif velocity_dist == 'custom':
		particle['v'][:] = vel
	else:
		particle['v'][:] = (0, 0, 0)
	particle['s'][:] = spin
	tc = 1.6e-4*mass/(k*temperature**8.0)
	particle['next_gas_coll_time'] = rng.exponential(particle['rng'], tc)
	particle['mass'] = mass
	particle['temperature'] = temperature
	particle['t'] = t0
	particle['x_old'] = particle['x'][:]
	particle['v_old'] = particle['v'][:]
	particle['gamma'] = gamma
	particle['tc'] = tc
	particle['coll_type'] = 0 #0 means no collision, 1 means wall, 2 means gas
	particle['wall_hit'] = 0 #0 means x, 1 means y, 2 means z
	particle['finished'] = 0 # is false, 1 is true
	particle['n_bounce'] = 0
	particle['n_coll'] = 0
	return particle