# Spin-tracking
New and improved version of the spin tracking code. This code is designed to run on either CPU or GPU.

## Installation

Installation of this code is done via git clone. We do not have a package manager installation or provide pre-compiled binary versions of the code base at this time. 

## Configuration and Compilation

This code is compiled via Makefile. Depending on the hardware you wish to target, you may wish to modify the makefile.

There are 3 main sets of flags that are defined in this makefile: one for CPU compilation, one for AMD GPU compilation with AMD HIP, and the final one for Nvidia GPU Compilation with Nvidia CUDA. You may not combine these at this time. There are examples for the various flags that are required for compilation presently in the Makefile. For those who just wish to target the CPU, leave all GPU related flags commented out and only uncomment the section labeled as being for the CPU.

For those who want to run this code on their GPU, comment out the section containing the relevant flags for your particular GPU brand. You will need to update any paths in these sections to match your Nvidia CUDA or AMD HIP installation. For Nvidia GPUs, it will also be necessary to identify the compute capability of your GPU. You can find that information here: https://en.wikipedia.org/wiki/CUDA#GPUs_supported. The value for SM should be the compute capability, just with the decimal place removed: 8.6 -> 86 for example. The Nvidia V100, A100, and H100 are 70, 80, and 90 respectively. For AMD GPUs this is not required, but note that not all AMD GPUs can be used with AMD ROCM/HIP. See this page for more details: https://docs.amd.com/bundle/Hardware_and_Software_Reference_Guide/page/Hardware_and_Software_Support.html. These are not restrictions of our code, but restrictions imposed by the manufacturers of that hardware so if your hardware is not compatible, apologies but we can't really do anything about that. 

### Modification of Field Gradients and Magnetic Fields
Constant electric field gradients and static magnetic field components can be modified via parameter file at this time. In addition, simple sinusoidal oscillating fields can be input via parameter file through control of the amplitude and driving frequency parameters. However, any field gradients or additional time dependence must be programmed into the code base directly and compiled before the code can be run. These modifications should be made in the src/integrator.cpp file, specifically the functions below.

- __PREPROC__ double3 pulse(const double t)
- __PREPROC__ double3 grad(double3& pos)

These functions expect only time and position input values respectively, but can be extended easily. These functions are used on both the CPU and GPU so only one function will need to be modified. Do NOT remove the __PREPROC__ flag as that specifies to the compiler to either compile for GPU or not. 

### Most Other Parameters
Most other parameters within this simulation code can be modified via a text parameter file. An example of one such file is included in params.txt. The following table shows the variables that can be directly modified at this time and their expected data type.

| Variable Name |  Description | Units | Data Type | Default Value |
| --- | --- | --- | --- | --- |
| B0 | Static background magnetic field | Tesla | double3 | 3.0e-6, 0.0, 0.0 |
| E | Static electric field | kV/cm | double3 | 0.0, 0.0, 75.0e5 |
| L | Size of the cell | meters | double3 | 0.07, 0.1, 0.4 |
| yi | Initial Spin Vector | na | double3 | 1.0, 0.0, 0.0 |
| m | Mass of Particle | kg | double | 1.20239e-26 |
| t0 | Start time of the simulation | seconds | double | 0.0 |
| tf | Stop time of the simulation | seconds |  double | 0.0 |
| rtol | Relative tolerance of the spin integration | NA | double | 1.0e-12 |
| rtol | Absolute tolerance of the spin integration | NA | double | 1.0e-12 |
| beta | DOP853 Spin-Integration parameter | NA | double | 0.0 |
| uround | DOP853 Spin-Integration parameter | NA | double | 1.0e-16 |
| safe | DOP853 Spin-Integration parameter | NA | double | 0.9 |
| fac1 | DOP853 Spin-Integration parameter | NA | double | 0.333 |
| fac2 | DOP853 Spin-Integration parameter | NA | double | 6.0 |
| hmax | Largest allowed step size of spin integration | seconds | double | 1.0 |
| hmin | Smallest allowed step size of spin integration | seconds | double | 1e-8 |
| h | Initial step size of spin integration | seconds | double | 0.001 |
| T | Temperature | Kelvin | double | 4.2 |
| gamma |  Gyromagnetic ratio of the particle | rad/s/T | double | -2.038e8 |
| V | Velocity of the particles if using dist='C' mode | m/s | double | 5.0 |
| a | Amplitude of the spin dressing field pulse | Tesla | double | 0.0 |
| w | Frequency of the spin dressing field pulse | Hz | double | 0.0 |
| maxPosStep | Maximum physical integrator step size | seconds | double | 0.1 |
| ioutInt | how frequently to output the state data | second | double | 0.05 |
| nmax | maximum number of iterations before an error is returned | NA | unsigned integer | 10000000 |
| seed | Starting random number seed | NA | unsigned integer | 0 |
| integratorType | See below section for options | NA | integer | 0 |
| numParticles | number of particles | NA | integer | 1000 |
| numPerGPUBlock | how many particles to handle on each compute unit of a GPU simultaneously (max at 1024) | NA | integer | 128 |
| iout | output type for DOP853 (deprecated) | NA | int | 2 |
| diffuse | Percent change of diffuse wall collisions | percent (0=0%, 1=100%) | float | 0 |
| dist | distribution type for velocity : 'C' = continuous, 'M' = maxwell boltzman | NA | character | 'C' |
| output | output format, see below for details | char | 'A' |
| gas_coll | simulate phonon collisions | NA | bool | true |
| gravity | enable gravity? | bool | true |
| fixedStepSize | Keep spin integrator step size fixed? | NA | bool | false |
| keepStepSize | Remember the step size between calls to the spin integrator | NA | bool | false |

## Data Output
Data output is fixed to outputting each particles time, position, velocity, and spin components every ioutInt. The output file is comprised of a header at the start that contains all of the user passed parameters. See the options.h file for the exact definition of the structure that is saved for understanding how to read it. 

After this header, the time, position, velocity, and spin data is saved in a flat binary structure. Each of these variables is saved as 8-byte floating point values, so each particle has 10 8-byte values associated with it. 

There previously were other output formats but these are not supported at this time. An example function for reading this data in is available for Python in scripts/nedmSim.py. This function will read in both the header and particle data and return them to the user as numpy arrays. 

