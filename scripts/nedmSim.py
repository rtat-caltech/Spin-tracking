#This is just a collection of a few convenience scripts that can be used for reading in the data and performing basic tasks

import numpy as np #handling array objects
import os #doing the file seeking to find the right location in the file

def readInOutputFile(outputFilename):
    """
    This function reads in the output from the Spin Tracking Simulations.
    
    Input(s):
    ----------------------------------------------
    filename: the name of the file to be read
        Expects this to be a string
    
    Output(s):
    ----------------------------------------------
    parameters: np.ndarray
        A single element from a numpy structured array that contains the parameters
        To see all of the defined parameters, use parameters.dtype.names
        This is a record of all of the input parameters used for the spin tracking code
        
    data: np.ndarray
        Only the 'n' output format is supported currently, others have been removed
        
        This is a multi-element numpy structured array that contains the state of the system at the requested output times
        It is 2-dimensional in the following format: 
            data[i,j] where i is the particle number, j is the timestep
        Each datapoint contains the following data
            t: time of the output
            xx: x coordinate of the position
            xy: y coordinate of the position
            xz: z coordinate of the position
            vx: x coordinate of the velocity
            vy: y coordinate of the velocity
            vz: z coordinate of the velocity
            sx: x coordinate of the spin
            sy: y coordinate of the spin
            sz: z coordinate of the spin
    
    """
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
        ('hmin', np.float64),
        ('h', np.float64),
        ('T', np.float64),
        ('sqrtKT_m', np.float64),
        ('tc', np.float64),
        ('gamma', np.float64),
        ('V', np.float64),
        ('a', np.float64),
        ('w', np.float64),
        ('swapStepSize', np.float64),
        ('maxPosStep', np.float64),
        ('ioutInt', np.float64),
        ('nmax', np.uint32),
        ('seed', np.uint32),
        ('integratorType', np.int32),
        ('numParticles', np.int32),
        ('numPerGPUBlock', np.int32),
        ('iout', np.int32),
        ('numPhiBins', np.int32),
        ('numThetaBins', np.int32),
        ('diffuse', np.float32),
        ('dist', 'S', 1),
        ('output', 'S', 1),
        ('gas_coll', bool),
        ('gravity', bool),
        ('fixedStepSize', bool),
        ('keepStepSize', bool),
        ('pad', 'S', 6) #this is padding space just designed to fix the structure padding done in C++
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
            ('vx', '<f8'),
            ('vy', '<f8'),
            ('vz', '<f8'),
            ('sx', '<f8'),
            ('sy', '<f8'),
            ('sz', '<f8')])
        file.seek(0, 0)
        data = np.fromfile(file, dtype=outputDtype, offset=parameters.nbytes)
        numTimes = data.shape[0]//parameters['numParticles']
        numPer = int((parameters['tf']-parameters['t0'])/parameters['ioutInt'])+1
        data = data[:numTimes*parameters['numParticles']].reshape(-1, parameters['numParticles']).T #this returns the data in a little more convenient format, I think
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

def writeParameterFile(filename, runParameters):
    """
    This function writes a parameter file to be used by the code. 
    
    Inputs:
    -------------------------------------
    filename: string
        The name of the file to create
    
    runParameters: dictionary
        The parameters to set in the file. 
        See the documentation on gitlab for the various input parameters and their default values.
        This function does support an additional specification for particle type:
            'particle': 'neutron' -> defines a neutron
                'gamma' = '-1.832471850e8'
                'm' = '1.674927e-27'
            'particle': 'helium-3' -> defines a helium-3
                'gamma' = '-2.037947093e8'
                'm' = '1.20200437865e-26'
    
    Outputs:
    ------------------------------------
    None:
        A file is created, no output is returned to the user
    """
    with open(filename, 'w') as f:
        for key in runParameters:
            if key == 'particle':
                if runParameters['particle'] == 'neutron':
                    gamma = '-1.832471850e8'
                    m = '1.674927e-27'
                    f.write('gamma, '+gamma + '\n')
                    f.write('m, '+m + '\n')
                elif runParameters['particle'] == 'helium-3':
                    gamma = '-2.037947093e8'
                    m = '1.20200437865e-26'
                    f.write('gamma, '+gamma + '\n')
                    f.write('m, '+m)
            else:
                f.write(key + ', '+runParameters[key]+'\n')
    return

def shiftDatapoints(phis):
    """
    This function will shift a set of angular components to be within one period of each other and properly aligned. Effectively it handles the output of np.arctan2 and aligns them to make sure that there is no overflow on the periodic boundaries
    
    Inputs
    ------------------------
    phis: np.ndarray
        Expects this to be a 1 dimensional np.ndarray of angles in radians

    Outputs
    -----------------------
    correctedPhis: np.ndarray
        An array shifted so that the angles defined do not wrap around the periodic boundary conditions
        
    """
    shifted = np.copy(phis)
    initHist, bins = np.histogram(phis, bins = 1000) #calculate the initial histogram and then adjust based on the mode
    peakloc = np.argmax(initHist)
    location = bins[peakloc]
    #shift everything to be close to that peak
    lowLocs = shifted<=(location-np.pi)
    highLocs = shifted>=(location+np.pi)
    shifted[lowLocs] += 2.0*np.pi
    shifted[highLocs] -= 2.0*np.pi
    mean = np.nanmean(shifted)
    std = np.nanstd(shifted)
    shifted[shifted<(mean-std*5)] = np.nan
    shifted[shifted>(mean+std*5)] = np.nan
    return shifted