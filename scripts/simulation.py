'''
This set of functions defines the actual particle tracking behavior.
'''

import numpy as np
import numba
import rng
import spinIntegrators as spin

__G_CONST = -9.8
__K_CONST = 1.380649e-23

def defineSimulation(cell= (0.07, 0.1, 0.4), t0 = 0.0, tf = 1.0, 
						 output_interval = 1.0, gas_coll = 1, diffuse = 1,
						 gravity = 1, max_step = 0.1, velocity_dist = 0, vel = 5,
						 temperature = 0.4, position_dist = 0) :
	"""
	This defines the parameters for the simulation itself.
	Parameters
	----------
		All inputs here are optional and have default values
		cell: (0.07, 0.1, 0.4)
			Cell size in meters for each dimension
		t0: 0.0
			start time of simulation in seconds
		tf: 1.0
			stop time of simulation in seconds
		output_interval = 1.0
			how frequently to output particle information in seconds
		gas_coll: 1
			0 means no gas collisions, 1 means gas collisions
		diffuse: 1
			0 means diffuse scattering is disabled, 1 means enabled
		gravity: 1
			If gravity is enabled. 0 means no gravity, 1 means gravity
		max_step: 0.1
			The largest step size allowed for the position tracking
		velocity_dist: 0
			0 means a fixed velocity, uses the input in vel to set the magnitude
			1 means maxwell boltzman distribution
		vel: 5
			The fixed velocity for all particles to use in m/s. Does not specify direction, only magnitude
			This is only used if velocity_dist == 0, otherwise ignored.
		temperature: 0.4
			The temperature of the system in kelvin
		position_dist: 0
			0 means a uniform random distribution within the cell
	Returns
	-------
	options structure
		numpy np.ndarray type containing the simulation options.
	"""
	dtype = np.dtype([
		('cell', np.float64, 3),
		('t0', np.float64),
		('tf', np.float64),
		('output_interval', np.float64),
		('gas_coll', np.int32),
		('diffuse', np.int32),
		('gravity', np.int32),
		('max_step', np.float64), 
		('velocity_dist', np.int32),
		('vel', np.float64),
		('temperature', np.float64),
		('position_dist', np.int32),
	])
	options = np.zeros(1, dtype=dtype)[0]
	options['cell'] = cell
	options['t0'] = t0
	options['tf'] = tf
	options['output_interval'] = output_interval
	options['gas_coll'] = gas_coll
	options['diffuse'] = diffuse
	options['gravity'] = gravity
	options['velocity_dist'] = velocity_dist
	options['vel'] = vel
	options['temperature'] = temperature
	options['position_dist'] = position_dist 
	options['max_step'] = max_step
	return options

def spinIntegratorOptions(integrator = 0, rtol = 1e-12, atol=1e-12, beta = 0.0, uround = 1e-16, safe = 0.9, fac1 = 0.333, fac2 = 6.0,
					 hmax = 1.0, h = 0.0001, nmax = 10000000, max_step = 0.1, min_step = 1.0e-9):
	"""
	This function creates the options structure that tells the spin integration functions how to behave.
	This does NOT control how the physics behaves, just how the system handles things like step sizes and whatnot.
	
	Parameters
	----------
		For all input parameter behavior, see the individual integration function
		integrator = 0
			The integration method. These are each defined in the spinIntegration function
			0: DOP853
				Adaptive step size DOP853 algorithm
			1: DOP853 with rotation matrices (known to be bugged)
				Adaptive step size DOP853 algorithm except using rotation matrices to update spin
			2: DOP853 with quaternions (known to be bugged)
				Adaptive step size DOP853 algorithm except using quaternions to update spin
			3: RK45
				Adaptive step size RK45 routine
			4: RK45 with rotation matrices
				Adaptive step size RK45 routine using rotation matrices to update spin
			5: RK45 with quaternions
				Adaptive step size RK45 routine using quaternions to update spin
			6: RK75109
				Adaptive step size RK75109 routine
			7: RK75109 with rotation matrices
				Adaptive step size RK75109 routine using rotation matrices to update spin
			8: RK75109 with quaternions (WIP)
				Adaptive step size RK75109 routine using quaternions to update spin (not implemented yet)
			9: Implicit Euler method
				Fixed step size implicit integrator. Work in progress to make adaptive
			10: Crank-Nicolson method
				Fixed step size implicit integrator. Work in progress to make adaptive
		All inputs here are optional.
		rtol: 1.0e-12
		atol: 1.0e-12
		beta: 0.0
		uround: 1.0e-16
		safe: 0.9
		fac1: 0.333
		fac2: 6.0
		hmax: 1.0
		h: 0.0001
		nmax: 10000000
		max_step: 0.1
		min_step: 1.0e-9
		
	Returns
	-------
	options structure
		numpy np.ndarray type containing the integrator options
	"""
	dtype = np.dtype([
		('integrator', np.int32),
		('rtol', np.float64),
		('atol', np.float64),
		('beta', np.float64),
		('uround', np.float64),
		('safe', np.float64),
		('fac1', np.float64),
		('fac2', np.float64),
		('hmax', np.float64),
		('nmax', np.uint32),
		('max_step', np.float64),
		('min_step', np.float64),
		('h', np.float64),
	])
	options = np.zeros(1, dtype=dtype)[0]
	options['integrator'] = integrator
	options['rtol'] = rtol
	options['atol'] = atol
	options['beta'] = beta
	options['uround'] = uround
	options['safe'] = safe
	options['fac1'] = fac1
	options['fac2'] = fac2
	options['hmax'] = hmax
	options['nmax'] = nmax
	options['max_step'] = max_step
	options['min_step'] = min_step
	options['h'] = h
	return options

particleDtype = np.dtype([
	('t', np.float64), #what time is it
	('x', np.float64, 3), #where is the particle now
	('v', np.float64, 3), #it's velocity
	('s', np.float64, 3), #it's spin
	('mass', np.float64), #it's mass
	('rng', np.uint64, 4), #rng seed
	('dt', np.float64), #the size of the next time step
	('t_old', np.float64), #previous time
	('x_old', np.float64, 3), #previous position
	('v_old', np.float64, 3), #previous velocity
	('gamma', np.float64), #gyromagnetic ratio
	('tc', np.float64), #collision time constant
	('next_gas_coll_time', np.float64), #next time for a gas collision
	('coll_type', np.int8), #collision type, 0 is no collision, 1 is wall, 2 is gas
	('wall_hit', np.int8), #which wall it hit
	('finished', np.int8), #is the simulation for this particle done
	('n_bounce', np.int64), #number of wall bounces
	('n_coll', np.int64), #number of collisions
	('n_steps', np.int64), #number of steps total
	('n_spin_steps', np.int64), #number of spin integration steps total
])

@numba.jit
def createParticle(simulationParameters, mass=2.2*5e-27, gamma = -2.078e8, spin=(1.0, 0.0, 0.0), seed=0):
	"""
	This function initializes a particle
	
	Parameters
	----------
	simulation_parameters
		The parameters of the simulation as created by the defineSimulation function
	mass
		The mass of the particle in kg
	gamma
		Gyromagnetic ratio of the particle in rad/s/Tesla
		Defaults to -2.078e8
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
	particle = np.zeros(1, dtype=particleDtype)[0]
	rng.xorshift128_init(seed, particle['rng']) #initialize the random numbers
	particle['t'] = simulationParameters['t0'] #set the start time
	#determine the starting position
	if simulationParameters['position_dist'] == 0:
		tx = rng.uniform(particle['rng'], -simulationParameters['cell'][0]/2.0, simulationParameters['cell'][0]/2.0)
		ty = rng.uniform(particle['rng'], -simulationParameters['cell'][1]/2.0, simulationParameters['cell'][1]/2.0)
		tz = rng.uniform(particle['rng'], -simulationParameters['cell'][2]/2.0, simulationParameters['cell'][2]/2.0)
		particle['x'][:] = (tx, ty, tz)
	else:
		particle['x'][:] = (0.0, 0.0, 0.0)
	#determine the starting velocity
	if simulationParameters['velocity_dist'] == 0:
		tx = rng.uniform(particle['rng'])
		ty = rng.uniform(particle['rng'])
		tz = rng.uniform(particle['rng'])
		tv = np.sqrt(tx**2.0 + ty**2.0 + tz**2.0)
		scale = simulationParameters['vel']/tv
		particle['v'][:] = (scale*tx, scale*ty, scale*tz)
	elif simulationParameters['velocity_dist'] == 1:
		sqrtKT_m = np.sqrt(__K_CONST*simulationParameters['temperature']/mass)
		tx = rng.normal(particle['rng'])*sqrtKT_m
		ty = rng.normal(particle['rng'])*sqrtKT_m
		tz = rng.normal(particle['rng'])*sqrtKT_m
		particle['v'][:] = (tx, ty, tz)
	else:
		particle['v'][:] = (0, 0, 0)
	particle['s'][:] = spin
	tc = 1.6e-4*mass/(__K_CONST*simulationParameters['temperature']**8.0)
	particle['next_gas_coll_time'] = rng.exponential(particle['rng'], tc)
	particle['mass'] = mass
	particle['x_old'] = particle['x'][:]
	particle['v_old'] = particle['v'][:]
	particle['gamma'] = gamma
	particle['tc'] = tc
	particle['coll_type'] = 0 #0 means no collision, 1 means wall, 2 means gas
	particle['wall_hit'] = 0 #0 means x, 1 means y, 2 means z
	particle['finished'] = 0 # is false, 1 is true
	particle['n_bounce'] = 0
	particle['n_coll'] = 0
	particle['t_old'] = particle['t']
	particle['dt'] = 0.0
	particle['n_steps'] = 0.0
	particle['n_spin_steps'] = 0
	return particle

@numba.jit
def sgn(val):
	if val < 0:
		return -1.0
	else:
		return 1.0

@numba.jit
def calc_next_collision_time(particle, options):
	"""
	This function calculates the next collision time for the particle.
	
	Parameters
	----------
	particle:
		A particle as created by the particle.createParticle function
	
	options:
		The simulation options as created by the simulation.simulationParameters function
	"""
	v = particle['v']
	x = particle['x']
	L = options['cell']
	coll_type=0 #0 is no collision
	wall_hit=0
	next_gas_coll_time = particle['next_gas_coll_time']
	n_bounce = particle['n_bounce']
	n_coll = particle['n_coll']
	t = particle['t']
	max_step = options['max_step']
	tf = options['tf']
	dt = 0
	dtx = 0
	dty = 0
	dtz = 0
	finished = False
	if options['gravity'] == 1:
		#do the calculations to figure out where the particle will hit next
		dx = sgn(v[0]) * L[0] / 2.0 - x[0]
		dz = sgn(v[2]) * L[2] / 2.0 - x[2];
		#time to wall for x and z coordinate
		dtx = dx / v[0];
		dtz = dz / v[2];
		y2 = v[1] * v[1];
		if sgn(v[1]) <= 0.0: #if the particle has negative y velocity
			dy = x[1] + L[1]*0.5;
			sqr = np.sqrt(-2.0*__G_CONST*dy+y2);
			temp1 = -(sqr+v[1])/__G_CONST;
			temp2 = (sqr-v[1])/__G_CONST;
			dty = min(abs(temp1), abs(temp2))
		else:
			maxHeight = -0.5 * y2/__G_CONST + x[1];
			if maxHeight < 0.5 * L[1]: #in this case it can't hit the ceiling
				dy = x[1]+L[1]*0.5;
				sqr = np.sqrt(-2.0*__G_CONST*dy+y2);
				temp1 = -(sqr+v[1])/__G_CONST;
				temp2 = (sqr-v[1])/__G_CONST;
				dty = max(temp1, temp2);
			else:
				dy = L[1]*0.5 - x[1]; #how far to ceiling
				sqr = np.sqrt(-2.0*__G_CONST*dy+y2);
				temp1 = -(sqr+v[1])/__G_CONST;
				temp2 = (sqr-v[1])/__G_CONST;
				dty = min(abs(temp1), abs(temp2));
		if dtx < 1e-16 or np.isnan(dtx):
			dtx = 1e6;
		elif dty < 1e-16 or np.isnan(dty):
			dty = 1e6;
		elif dtz < 1e-16 or np.isnan(dtz):
			dtz = 1e6;
	else:
		dx = sgn(v[0]) * L[0] / 2.0 - x[0];
		dy = sgn(v[1]) * L[1] / 2.0 - x[1];
		dz = sgn(v[2]) * L[2] / 2.0 - x[2];
		
		dtx = dx / v[0];
		dty = dy / v[1];
		dtz = dz / v[2];
		if (dtx < 1e-16):
			dtx = 1e6;
		elif (dty < 1e-16):
			dty = 1e6;
		elif (dtz < 1e-16):
			dtz = 1e6;
	min_elm = -1
	if dtx <= dty and dtx <= dtz:
		tbounce = dtx;
		min_elm = 0;
	elif dty <= dtx and dty <= dtz:
		tbounce = dty;
		min_elm = 1;
	elif dtz <= dtx and dtz <= dty:
		tbounce = dtz;
		min_elm = 2;
	timeToNextGas = next_gas_coll_time - t;
	if max_step <= timeToNextGas and max_step <= tbounce and t + max_step < tf: #check if the max step size is smaller than the next collision times
		#if so then just say we don't collide and keep going
		dt = max_step;
		coll_type = 0
	elif tbounce < timeToNextGas and t + tbounce < tf: #is a wall bounce next
		dt = tbounce;
		n_bounce += 1;
		coll_type = 1
		wall_hit = min_elm #0 is x, 1 is y, 2 is z
	elif t + tbounce > next_gas_coll_time and next_gas_coll_time < tf: #is a gas collision next?
		dt = next_gas_coll_time - t;
		next_gas_coll_time += rng.exponential(particle['rng'], particle['tc']);
		coll_type = 2
		n_coll += 1;
	else: #in this case it reached the end of the simulation
		coll_type = 0
		dt = tf - t;
		finished = True;
	particle['finished'] = finished
	particle['wall_hit'] = wall_hit
	particle['coll_type'] = coll_type
	particle['next_gas_coll_time'] = next_gas_coll_time
	particle['n_bounce'] = n_bounce
	particle['n_coll'] = n_coll
	particle['dt'] = dt
	return particle

@numba.jit
def new_velocities(particle, options):
	v = particle['v'][:]
	particle['v_old'] = v[:]
	Vel = np.sqrt(v[0]**2.0+v[1]**2.0+v[2]**2.0);
	if particle['coll_type'] == 0:
		return particle
	elif particle['coll_type'] == 0 and options['diffuse'] == 0:
		if particle['wall_hit'] == 0:
			v[0] *= -1.0;
		elif particle['wall_hit'] == 1:
			v[1] *= -1.0;
		elif particle['wall_hit'] == 2:
			v[2] *= -1.0;
	elif particle['coll_type'] == 1 and options['diffuse'] == 1:
		phi = np.arccos(np.sqrt(rng.uniform(particle['rng'])));
		theta = rng.uniform(particle['rng'])*np.pi*2.0;
		if particle['wall_hit'] == 0:
			v[0] = -1 * sgn(v[0]) * Vel * np.cos(phi);
			v[1] = -Vel * np.sin(phi) * np.cos(theta);
			v[2] = Vel * np.sin(phi) * np.sin(theta);
		elif particle['wall_hit'] == 1:
			v[0] = Vel * np.sin(phi) * np.cos(theta);
			v[1] = -1 * sgn(v[1]) * Vel * np.cos(phi);
			v[2] = Vel * np.sin(phi) * np.sin(theta);
		elif particle['wall_hit'] == 2:
			v[0] = Vel * np.sin(phi) * np.cos(theta);
			v[1] = Vel * np.sin(phi) * np.sin(theta);
			v[2] = -1 * sgn(v[2]) * Vel * np.cos(phi);
	elif particle['coll_type'] == 2 and options['velocity_dist'] == 1:
		sqrtKT_m = np.sqrt(__K_CONST*options['temperature']/particle['mass'])
		v[0] = rng.normal(particle['rng'])*sqrtKT_m
		v[1] = rng.normal(particle['rng'])*sqrtKT_m
		v[2] = rng.normal(particle['rng'])*sqrtKT_m
	elif particle['coll_type'] == 2 and options['velocity_dist'] == 0:
		tx = rng.normal(particle['rng'])
		ty = rng.normal(particle['rng'])
		tz = rng.normal(particle['rng'])
		vec_norm = np.sqrt(tx**2.0 + ty**2.0 + tz**2.0)
		scale = Vel/vec_norm
		v[0] = tx*scale
		v[1] = ty*scale
		v[2] = tz*scale
	particle['v'] = v
	return particle

@numba.jit
def move(particle, options):
	particle['t_old'] = particle['t'] #update the old time to the now time
	particle['t'] += particle['dt'] #update now time
	particle['x_old'] = particle['x'][:] #update the old position
	particle['v_old'] = particle['v'][:] #update the old velocity
	particle['x'][:] = particle['x_old'] +  particle['v'] * particle['dt']
	if options['gravity'] == 1:
		particle['x'][1] += 0.5 *__G_CONST*particle['dt']*particle['dt']
		particle['v'][1] += __G_CONST*particle['dt']
	return particle

@numba.jit
def step(particle, simulation, spinOptions, BField, EField):
	particle = calc_next_collision_time(particle, simulation)
	particle = move(particle, simulation)
	particle = new_velocities(particle, simulation)
	particle = spin.integrateSpin(particle, simulation, spinOptions, BField, EField)
	particle['n_steps']+=1
	return particle

@numba.jit
def run(particle, simulation, spinOptions, BField, EField):
	while particle['finished'] == 0:
		particle = step(particle, simulation, spinOptions, BField, EField)
	return particle