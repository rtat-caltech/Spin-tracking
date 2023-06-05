"""
This script reads in a single .out file from the simulations and compresses it into HDF5.

It only works for the output type 'n' so don't try this with other output types.
"""

import numpy as np #handling array objects
import os #doing the file seeking to find the right location in the file
import h5py
import sys

def readInOutputFile(outputFilename):
	optionsStructType = np.dtype([
		('B0', np.float64, 3),
		('E', np.float64, 3),
		('L', np.float64, 3),
		('yi', np.float64, 3),
		('posHistBins', np.float64, 3),
		('m', np.float64),
		('t0', np.float64),
		('tf', np.float64),
		('rtol', np.float64),
		('atol', np.float64),
		('beta', np.float64),
		('uround', np.float64),
		('safe', np.float64),
		('fac1', np.float64),
		('fac2', np.float64),
		('hmax', np.float64),
		('h', np.float64),
		('T', np.float64),
		('gamma', np.float64),
		('V', np.float64),
		('swapStepSize', np.float64),
		('ioutInt', np.float64),
		('nmax', np.uint32),
		('integratorType', np.int32),
		('numParticles', np.int32),
		('numPerGPUBlock', np.int32),
		('iout', np.int32),
		('numPhiBins', np.int32),
		('numThetaBins', np.int32),
		('dist', 'S', 1),
		('output', 'S', 1),
		('gas_coll', bool),
		('diffuse', bool),
		('gravity', bool),
		('pad', 'S', 7) #this is padding space just designed to fix the structure padding done in C++
	])
	file = open(outputFilename, 'rb')
	parameters = np.fromfile(file, count=1, dtype=optionsStructType)[0]
	data = None
	if parameters['output'] == b'n':
		#in this case it was the full dump of all particle data
		outputDtype = np.dtype([
			('t', '<f8'),
			('xx', '<f8'),
			('xy', '<f8'),
			('xz', '<f8'),
			('sx', '<f8'),
			('sy', '<f8'),
			('sz', '<f8')])
		file.seek(0, 0)
		data = np.fromfile(file, dtype=outputDtype, offset=parameters.nbytes)
		numPer = int((parameters['tf']-parameters['t0'])/parameters['ioutInt'])+1
		data = data.reshape(numPer, parameters['numParticles']).T #this returns the data in a little more convenient format, I think
	elif parameters['output'] == b'h':
		#this is the histogram output format instead now
		numx = int(parameters['L'][0]/parameters['gridSize'])+1
		numy = int(parameters['L'][1]/parameters['gridSize'])+1
		numz = int(parameters['L'][2]/parameters['gridSize'])+1
		
		numVecBins = int(np.ceil(2.0/parameters['vecBinSize']))
		
		xbins = np.arange(-parameters['L'][0]/2.0, parameters['L'][0]/2.0, parameters['gridSize'])
		ybins = np.arange(-parameters['L'][0]/2.0, parameters['L'][0]/2.0, parameters['gridSize']) 
		zbins = np.arange(-parameters['L'][0]/2.0, parameters['L'][0]/2.0, parameters['gridSize'])
		sbins = np.arange(-1.0, 1.0, parameters['vecBinSize'])
		outputDtype = np.dtype([
			('t', np.float64),
			('x', np.uint32, numx),
			('y', np.uint32, numy),
			('z', np.uint32, numz),
			('sx', np.uint32, numVecBins),
			('sy', np.uint32, numVecBins),
			('sz', np.uint32, numVecBins)
		])
		file.seek(0, 0)
		data = np.fromfile(file, dtype=outputDtype, offset=parameters.nbytes)
		data = {'xbins': xbins,
			   'ybins': ybins,
			   'zbins': zbins,
			   'sbins': sbins,
			   'data': data}
	elif parameters['output'] == b'a':
		outputDtype = np.dtype([
			('t', np.float64),
			('sx', np.float64),
			('dsx', np.float64),
			('sy', np.float64),
			('dsy', np.float64),
			('sz', np.float64),
			('dsz', np.float64)
		])
		file.seek(0, 0)
		data = np.fromfile(file, dtype=outputDtype, offset=parameters.nbytes)
	file.close()
	return parameters, data

def modifyData(data, params):
	outputDtype = np.dtype([
			('t', np.float32),
			('x', np.uint8),
			('y', np.uint8),
			('z', np.uint8),
			('sx', np.uint16),
			('sy', np.uint16),
			('sz', np.uint16)
		])
	outputData = np.zeros(data.shape, dtype=outputDtype)
	outputData['t'] = data['t']
	outputData['x'] = (data['xx']+params['L'][0]/2.0)/params['L'][0]*2**8
	outputData['y'] = (data['xy']+params['L'][1]/2.0)/params['L'][1]*2**8
	outputData['z'] = (data['xz']+params['L'][2]/2.0)/params['L'][2]*2**8
	outputData['sx'] = (data['sx']+1.0)/2.0*2**16
	outputData['sy'] = (data['sy']+1.0)/2.0*2**16
	outputData['sz'] = (data['sz']+1.0)/2.0*2**16
	return outputData

argv = sys.argv

if len(argv) != 3:
	print('invalid input: expects input_name output_name')
	sys.exit()

inputname = argv[1]
outputname = argv[2]

parameters, data = readInOutputFile(inputname)
test = modifyData(data, parameters)

output = h5py.File(outputname, 'w')
output.create_dataset('parameters', data=parameters)
output.create_dataset('data', data=test, compression='gzip')
output.close()