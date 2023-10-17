#This is just a collection of a few convenience scripts that can be used for reading in the data and performing basic tasks

import numpy as np #handling array objects
import os #doing the file seeking to find the right location in the file

def readInOutputFile(outputFilename, prec=np.float64, pad = 0):
    """
    This function reads in the output from the Spin Tracking Simulations.
    
    Input(s):
    ----------------------------------------------
    filename: the name of the file to be read
        Expects this to be a string
       
    prec: np.dtype
        The precision of the floating point values
        
    pad: int
        The number of bytes of padded to round the structure to the proper padded C length
    
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
        ('B0', prec, 3),
        ('E', prec, 3),
        ('L', prec, 3),
        ('yi', prec, 3),
        ('m', prec),
        ('t0', prec),
        ('tf', prec),
        ('rtol', prec),
        ('atol', prec),
        ('beta', prec),
        ('uround', prec),
        ('safe', prec),
        ('fac1', prec),
        ('fac2', prec),
        ('hmax', prec),
        ('hmin', prec),
        ('h', prec),
        ('T', prec),
        ('sqrtKT_m', prec),
        ('tc', prec),
        ('gamma', prec),
        ('V', prec),
        ('a', prec),
        ('w', prec),
        ('swapStepSize', prec),
        ('maxPosStep', prec),
        ('ioutInt', prec),
        ('diffuse', prec),
        ('nmax', np.uint32),
        ('seed', np.uint32),
        ('integratorType', np.int32),
        ('numParticles', np.int32),
        ('numPerGPUBlock', np.int32),
        ('iout', np.int32),
        
        ('dist', 'S', 1),
        ('gas_coll', bool),
        ('gravity', bool),
        ('fixedStepSize', bool),
        ('keepStepSize', bool),
        ('pad', 'S', pad) #this is padding space just designed to fix the structure padding done in C++
    ])
    file = open(outputFilename, 'rb')
    parameters = np.fromfile(file, count=1, dtype=optionsStructType)[0]
    data = None
    #in this case it was the full dump of all particle data
    outputDtype = np.dtype([
        ('t', prec, parameters['numParticles']),
        ('x', prec, (parameters['numParticles'], 3)),
        ('v', prec, (parameters['numParticles'], 3)),
        ('s', prec, (parameters['numParticles'], 3)),
        ('errorState', np.int32, parameters['numParticles']),
        ('n_coll', np.int64, parameters['numParticles']),
        ('n_bounce', np.int64, parameters['numParticles']),
        ('n_steps', np.int64, parameters['numParticles'])
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