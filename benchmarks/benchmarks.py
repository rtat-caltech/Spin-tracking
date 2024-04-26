import numpy as np
import subprocess
import re
import os
import itertools
import h5py
from argparse import ArgumentParser

kB = 1.3806e-23 # J/K
c = 3e8 # m/s
default_config = """
dist, M
T, 0.35
gas_coll, true
diffuse, 0.0
rtol, 1.0e-12
gravity, false
B0, 3.0e-6, 0, 0
h, 0.0001
numPerGPUBlock, 256
t0, 0.0
tf, 10.0
a, 3.0e-5
w, 6.28318530718e3
E, 0, 0, 0
numParticles, 65536
Gz, 0, 0, 3e-9
ioutInt, 1.0
integratorType, 0
"""

def modify_parameter(configuration, param, value):
    if not isinstance(value, str):
        value = str(value)
    new_config, nsub = re.subn(r'^({:s}), (.*)$'.format(param), r'\1, {:s}'.format(value), configuration, flags=re.MULTILINE, count=1)
    if nsub > 0:
        return new_config
    else:
        return configuration + '\n{:s}, {:s}'.format(param, value)

def string_to_integrator(s):
    mapping = {r'dop\d*$' : 0,
               r'rk\d*$' : 1,
               r'magnus.*$' : 2,
               r'rk\d*q.*$' : 3,
               r'rkf\d*$' : 4,
               r'rkf\d*q.*$' : 5,
               r'floquet.*g.*$' : 6,
               r'floquet.*fft.*$' : 7}

    for key in mapping:
        if re.match(key, s):
            return mapping[key]
        
    if re.match(r'\d+$', s):
        res = int(s)
        if res >= 0 and res <= 8:
            return res

    raise ValueError('Unrecognized integrator type {:s}'.format(s))

def run(executable, config_file, out_file):
    subprocess.call([executable, config_file, out_file])

def run(executable, scratch_dir, config_string, out_file_name):
    config_file = os.path.join(scratch_dir, '_config_tmp.txt')
    with open(config_file, 'w') as f:
        f.write(config_string)
    run(executable, config_file, os.path.join(scratch_dir, out_file_name))

def delete_file(file_to_delete):
    if os.path.isfile(file_to_delete):
        os.remove(file_to_delete)
    else:
        print('Warning: file {:s} does not exist'.format(file_to_delete))

def main():
    parser = ArgumentParser(prog='Script for Running Benchmarks')
    parser.add_argument('main', help='location of executable')
    parser.add_argument('scratch', help='directory to put output files')
    parser.add_argument('integrator', type=string_to_integrator)
    parser.add_argument('--clean', help='delete generated output files')
    
    args = parser.parse_args()

    if os.path.isfile(args.main):
        executable = args.main
    else:
        raise ValueError('Executable {:s} does not exist'.format(args.scratch))

    if os.path.isdir(args.scratch):
        out_dir = args.scratch
    else:
        raise ValueError('Scratch directory {:s} does not exist'.format(args.scratch))

    config_file = os.path.join(out_dir, '_config_tmp.txt')
    counter = 0
    for T in np.linspace(0.3, 0.5, num=3):
        config = modify_parameter(default_config, 'T', '{:.2f}'.format(T))
        config = modify_parameter(config, 'integratorType', args.integrator)
        
        with open(config_file, 'w') as f:
            f.write(config)

        output_name = 'output_{:d}.hdf5'.format(counter)
        counter += 1
        print('Running with ')
        run(executable, config_file, os.path.join(out_dir, output_name))


    print('Cleaning')
    delete_file(config_file)
    if args.clean:
        delete_file(output_file)

def get_config_string(param_dict):
    return '\n'.join(["{:s}, {:s}".format(k, v) for (k, v) in items])

class Benchmark:    
    def __init__(self, physics, integration, executable, scratch_dir='.'):
        self.config = dict()
        self.config.update(physics)
        self.config.update(integration)
        self.executable = executable
        self.scratch_dir = scratch_dir

    def __init__(self, scratch_dir='.'):
        #If you only want to analyze pre-existing data 
        self.scratch_dir = scratch_dir

    def run(self):
        pass

    def run(self, out_file_name):
        run(self.executable, self.scratch_dir, get_config_string(self.config), out_file_name)

    def analyze(self):
        pass

    def clean(self):
        pass

class PerformanceBenchmark(Benchmark):
    def __init__(self, physics, integration, executable, scratch_dir):
        super().__init__(self, physics, integration, executable, scratch_dir)

    def run(self):
        out_file_name = 'perf_out.hdf5'
        t1 = time.time()
        super().run(self, out_file_name)
        t2 = time.time()
        self.elapsed_time = t2 - t1

    def analyze(self):
        print('Total elapsed time: {:.3f}'.format(self.elapsed_time))

def knudsen_and_adiabaticity(cell_dims, temperature, mass, w0):
    # Returns Knudsen number and adiabaticity
    vT = np.sqrt(kB * temperature/mass)
    tc = 1.6e-4 * mass/(kB * temperature**8.0)
    mean_free_path = vT * tc
    knudsen_number = mean_free_path/min(cell_dims) # <<1 : diffusive
    D = vT * vT/tc
    if knudsen_number < 1:
        # Diffusive regime
        adiabaticity = w0 * R * R/D
    else:
        adiabaticity = w0 * mean_free_path**2 / vT

    if abs(knudsen_number - 1) < 0.5:
        print('Warning: you might be in the transition regime (Knusden number close to 1)')

    print('Regime: ' + ('diffusive ' if knudsen_number < 1 else 'ballistic ')
          + ('adiabatic' if adiabaticity > 1 else 'non-adiabatic'))
    
    return knudsen_number, adiabaticity

class PignolBenchmark(Benchmark):
    # Allows you to compare to the Pignol result
    def __init__(self, physics, integration, executable, scratch_dir,
                 Gxx_range=[0], Gyy_range=[0], Gzz_range=[0]
                 Gxy_range=[0], Gyz_range=[0], Gzy_range=[0],
                 E_range=[0]):
        super().__init__(self, physics, integration, executable, scratch_dir)
        self.Gxx_range = Gxx_range # Array of dBx/dx values
        self.Gyy_range = Gyy_range # Array of dBy/dy values
        self.Gzz_range = Gzz_range # Array of dBz/dz values
        self.Gxy_range = Gyz_range # Array of dBx/dy values (dBy/dx constrained by Maxwell's)
        self.Gyz_range = Gyz_range # Array of dBy/dz values (dBz/dy constrained by Maxwell's)
        self.Gzx_range = Gzx_range # Array of dBz/dx values (dBx/dz constrained by Maxwell's)
        self.E_range = E_range # Array of Ex values

        self.output_files = []
        
        cell_dims = parse_triplet(self.config['L'])
        temperature = float(self.config['T'])
        mass = float(self.config['m'])
        w0 = parse_triplet(self.config['B0'])[0] * float(self.config['gamma'])
        self.knudsen_number, self.adiabaticity = knudsen_and_adiabaticity(
            cell_dims, temperature, mass, w0
        )

    def __init__(self):
        pass

    def run(self):
        field_iterator = itertools.product(self.Gx_range, self.Gy_range, self.Gz_range)
        counter = 0
        for params in field_iterator:
            Gxx, Gyy, Gzz, Gxy, Gyz, Gzx, E  = params
            self.config['Gx'] =  '{:.2e}, {:.2e}, {:.2e}'.format(Gxx, Gxy, Gxz)
            self.config['Gy'] = '{:.2e}, {:.2e}, {:.2e}'.format(-Gxy, Gyy, Gyz)
            self.config['Gz'] = '{:.2e}, {:.2e}, {:.2e}'.format(Gzx, -Gyz, Gzz)
            out_file_name = 'out_{:d}.hdf5'.format(counter)
            self.output_files.append(os.path.join(self.scratch_dir, out_file_name))
            super().run(self, out_file_name)
            
    def analyze(self, files=None):
        if files is None:
            files = self.output_files
        counter = 0
        for fname in files:
            with h5py.File(fname, 'r') as f:
                gamma = f.attrs['gamma']
                B0 = f.attrs['B0'][0]
                cell_dims = np.array(f.attrs['L'])
                Gxx, Gxy, Gxz = f.attrs['Gx']
                Gyx, Gyy, Gyz = f.attrs['Gy']
                Gzx, Gzy, Gzz = f.attrs['Gz']
                E = f.attrs['E']
                w0 = f.attrs['gamma'] * B0
                G = np.array([Gxx, Gxy, Gxz;
                              Gyx, Gyy, Gyz;
                              Gzx, Gyz, Gzz])
                
                if counter == 0 and len(self.output_files) == 0:
                    temperature = f.attrs['T']
                    mass = f.attrs['m']
                    self.knudsen_number, self.adiabaticity = knudsen_and_adiabaticity(
                        cell_dims, temperature, mass, w0
                    )
                counter += 1
                L2 = np.power(np.array(cell_dims), 2)
                bx2, by2, bz2 = G @ L2
                if self.adiabaticity > 1:
                    expected_B2 = gamma**2/(2 * w0) * (by2 + bz2) + \
                        gamma**2/(6 * w0**3) * 3 * vT**2 * np.sum(G[1,:] * G[1,:]) + np.sum(G[2,:] * G[2,:])
                    expected_E2 = gamma**2 * E**2/(3 * c**4 * w0) * 3 * vT**2
                    expected_BE = -gamma**2 * E/(c**2 * w0**2) * (Gyy * vT**2 + Gzz * vT**2)
                else:
                    expected_B2 = 0 # Need an analytic expression for that integral
                    expected_E2 = -gamma**2 * E**2/(2 * c**4) * (L2[1] + L2[2])/12
                    expectd_BE = gamma*2 * E/c**2 * (Gyy * L2[1] + Gzz * L2[2])/12
                
                total_shift = expected_B2 + expected_E2 + expected_BE
                spin_data = np.array(f['Spin'])
                b_end = np.mean(spin_data[-1,:,:], axis=0)
                
        return

def parse_triplet(s):
    m = re.match('(.*),(.*),(.*)', s)
    return (float(m.group(1)), float(m.group(2)), float(m.group(3)))
    

if __name__ == "__main__":
    main()

