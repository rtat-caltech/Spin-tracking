# Spin-tracking-New
New and improved version of the spin tracking code. This code is designed to run on either CPU or GPU.

## Installation

Installation of this code is done via git clone. We do not have a package manager installation or provide pre-compiled binary versions of the code base at this time. 

## Configuration and Compilation

This code is compiled via Makefile. Depending on the hardware you wish to target, you may wish to modify the makefile.

There are 3 main sets of flags that are defined in this makefile: one for CPU compilation, one for AMD GPU compilation with AMD HIP, and the final one for Nvidia GPU Compilation with Nvidia CUDA. You may not combine these at this time. There are examples for the various flags that are required for compilation presently in the Makefile. For those who just wish to target the CPU, leave all GPU related flags commented out and only uncomment the section labeled as being for the CPU.

For those who want to run this code on their GPU, comment out the section containing the relevant flags for your particular GPU brand. You will need to update any paths in these sections to match your Nvidia CUDA or AMD HIP installation. For Nvidia GPUs, it will also be necessary to identify the compute capability of your GPU. You can find that information here: https://en.wikipedia.org/wiki/CUDA#GPUs_supported. The value for SM should be the compute capability, just with the decimal place removed: 8.6 -> 86 for example. For AMD GPUs this is not required, but note that not all AMD GPUs can be used with AMD ROCM/HIP. See this page for more details: https://docs.amd.com/bundle/Hardware_and_Software_Reference_Guide/page/Hardware_and_Software_Support.html. These are not restrictions of our code, but restrictions imposed by the manufacturers of that hardware so if your hardware is not compatible, apologies but we can't really do anything about that. 

### Modification of Field Gradients and Magnetic Fields
Constant electric field gradients and static magnetic field components can be modified via parameter file at this time. However, any field gradients or time dependent field gradients must be programmed into the code base directly and compiled before the code can be run. These modifications should be made in the src/integrator.cpp file, specifically the functions below.

- __PREPROC__ double3 pulse(const double t)
- __PREPROC__ double3 grad(double3& pos)

Do not adjust the arguments passed into the function as that will break the code (without further modification), just modifiy the code within the function. If your modification requires more inputs or adjustments, let a developer know and we will find a solution for you.

### Most Other Parameters
Most other parameters within this simulation code can be modified via a text parameter file. An example of one such file is included in params.txt. The following table shows the variables that can be directly modified at this time and their expected data type.

| Variable Name |  Description | Data Type | Default Value |
| --- | --- | --- | --- |
| B0 | Static background magnetic field | double3 | 0.0, 0.0, 3.0e-6 |
| E | Static electric field | double3 | 0.0, 0.0, 75.0e5 |
| L | Size of the cell (m) | double3 | 0.07, 0.1, 0.4 |
| yi | Initial Spin Vector | double3 | 1.0, 0.0, 0.0 |
| m | Mass of Particle (kg) | double | 2.2*5e-27 |
| t0 | Start time of the simulation (s) | double | 0.0 |
| tf | Stop time of the simulation (s) | double | 0.0 |
| rtol | Relative tolerance of the spin integration | double | 1.0e-12 |
| rtol | Absolute tolerance of the spin integration | double | 1.0e-12 |
| beta | DOP853 Spin-Integration parameter | double | 0.0 |
| uround | DOP853 Spin-Integration parameter | double | 1.0e-16 |
| safe | DOP853 Spin-Integration parameter | double | 0.9 |
| fac1 | DOP853 Spin-Integration parameter | double | 0.333 |
| fac2 | DOP853 Spin-Integration parameter | double | 6.0 |
| hmax | Largest allowed step size  of spin integration (s) | double | 1.0 |
| h | Initial step size of spin integration (s) | double | 0.001 |
| tc | Collision time constant | double | 1.0e-4 |
| T | Temperature (K) | double | 4.2 |
| gamma |  Gyromagnetic ratio of the particle (rad/s/T) | double | -2.078e8 |
| V | Velocity of the neutrons (m/s) | double | 5.0 |
| swapStepSize | Step size above which euler-rodriguez formula is used instead (s) | double | 1.0e-4 |
| gridSize | Output histogram size (cm) | double | 1.0 |
| vecBinSize | width of each bin for spin data (0->1) | double | 0.1 |
| ioutInt | how frequently to output the state data (s) | double | 0.05 |
| nmax | maximum number of iterations before an error is returned | unsigned integer | 10000000 |
| integratorType | 0 means DOP853, 1 means hybrid rk45, 2 means no spin integration | integer | 0 |
| numParticles | number of particles | integer | 1000 |
| numPerGPUBlock | how many particles to handle on each compute unit of a GPU simultaneously (max at 1024) | integer | 128 |
| iout | output type (deprecated) | int | 2 |
| dist | distribution type for velocity : 'C' = continuous, 'M' = maxwell boltzman | character | 'C' |
| output | output format, see below for details | char | 'A' |
| gas_coll | simulate gas collisions? | bool | true |
| diffuse | enable diffuse wall collisions? | bool | true |
| gravity | enable gravity? | bool | true |

## Data Output
Data output is controlled via the 'output' parameter in the input parameter file. This parameter expects to be set to either 'A', 'N', or 'H'. For information about how to read in these files, see the provided Jupyter Notebook OutputTesting.ipynb. That contains an example of how to read the output data file for each version of the input parameter. 

Output 'N' saves all of the particle data during the simulation as often as the user requests it, determined by the ioutInt parameter. This means the particle position and spin information are all saved to the output file, for every particle. This can result in very large output files so use at your own risk.

Output 'H' saves a pair of histograms of the particle state data at each time interval. The first histogram is for the particle position data. This will contain a 3 histograms, one for each coordinate within the cell, with a bin size determined by the 'gridSize' parameter. The second set of 3 histograms if for the spin state data that contains bins with a size set by 'vecBinSize'. 

Output 'A' only saves the average spin vector components and their standard deviations for every particle in the system at each time interval. This is the smallest output file, but also the least informative. 

