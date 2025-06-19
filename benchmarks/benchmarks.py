import numpy as np
import subprocess
import re
import os
import itertools
import h5py
import time
import pandas as pd
import sys
from scipy.special import jv, erfcx
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
class Collector:
    def __init__(self):
        self.rows = []
    def new_record(self):
        self.rows.append(dict())
    def add(self, field_name, data):
        self.rows[-1][field_name] = data
    def to_pandas(self):
        return pd.DataFrame(self.rows)

def modify_parameter(configuration, param, value):
    if not isinstance(value, str):
        value = str(value)
    new_config, nsub = re.subn(r'^({:s}), (.*)$'.format(param), r'\1, {:s}'.format(value), configuration, flags=re.MULTILINE, count=1)
    if nsub > 0:
        return new_config
    else:
        return configuration + '\n{:s}, {:s}'.format(param, value)

def integrator_to_string(d):
    mapping = {0 : 'DOP',
               1: 'RK4',
               2: 'Magnus CFET',
               3: 'RK4 Quaternion',
               4: 'RKF4',
               5: 'RKF4 Quaternion',
               6: 'Floquet Goertzel',
               7: 'Floquet FFT'}
    return mapping[d]

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

def run(executable, scratch_dir, config_string, out_file_name):
    config_file = os.path.join(scratch_dir, '_config_tmp.txt')
    with open(config_file, 'w') as f:
        f.write(config_string)
        
    process = subprocess.Popen([executable, config_file, os.path.join(scratch_dir, out_file_name)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, env=os.environ.copy())
    while True:
        line = process.stdout.readline()
        if process.poll() != None:
            break
        if line != '':
            sys.stdout.write(line)
            sys.stdout.flush()
    out, err = process.communicate()
    print(out, err)
    return out, err

def delete_file(file_to_delete):
    if os.path.isfile(file_to_delete):
        os.remove(file_to_delete)
    else:
        print('Warning: file {:s} does not exist'.format(file_to_delete))

def get_config_string(param_dict):
    return '\n'.join(["{:s}, {:s}".format(k, v) for (k, v) in param_dict.items()])

def parse_triplet(s):
    m = re.match('(.*),(.*),(.*)', s)
    return (float(m.group(1)), float(m.group(2)), float(m.group(3)))

class Benchmark:    
    def __init__(self, physics=dict(), integration=dict(), executable=None, scratch_dir='.'):
        self.config = dict()
        self.config.update(physics)
        self.config.update(integration)
        self.executable = executable
        if not os.path.isdir(scratch_dir):
            os.mkdir(scratch_dir)
        self.scratch_dir = scratch_dir
        self.rand_seed = 'seed' in integration and integration['seed'] == 'rand'

    def run(self):
        pass

    def run(self, out_file_name):
        if not os.path.isfile(self.executable):
            raise ValueError('Executable {:s} does not exist'.format(self.executable))
        if self.rand_seed:
            self.config['seed'] = str(np.random.randint(0, 2**16))
        run(self.executable, self.scratch_dir, get_config_string(self.config), out_file_name)

    def analyze(self):
        pass

    def clean(self):
        pass

class PerformanceBenchmark(Benchmark):
    def __init__(self, physics=dict(), integration=dict(), executable=None, scratch_dir='.',
                integrators=[0], n_range=[0], a_range=[0]):
        # integrators: list of integrators to try
        # n_range: list of particle range
        # a_range: dressing field strength range
        super().__init__(physics=physics, integration=integration, executable=executable, scratch_dir=scratch_dir)
        self.integrators = integrators
        self.n_range = n_range
        self.a_range = a_range
        self.output_files = []
        
    def run(self):
        field_iterator = itertools.product(self.integrators, self.n_range, self.a_range)
        for params in field_iterator:
            integrator, n_part, a  = params
            self.config['a'] =  '{:.2e}'.format(a)
            self.config['numParticles'] = '{:d}'.format(n_part)
            self.config['integratorType'] = '{:d}'.format(integrator)
            out_file_name = 'out_{:d}.hdf5'.format(int(time.time() * 1000))
            self.output_files.append(os.path.join(self.scratch_dir, out_file_name))
            super().run(out_file_name)

    def analyze(self, files=None):
        if files is None:
            files = self.output_files
        col = Collector()
        for fname in files:
            with h5py.File(fname, 'r') as f:
                col.new_record()
                col.add('a', f.attrs['a'])
                col.add('n', f.attrs['numParticles'])
                col.add('iType', f.attrs['integratorType'])
                col.add('sim time', f.attrs['Simulation Time'])
        return col.to_pandas()

def knudsen_and_adiabaticity(cell_dims, temperature, mass, w0, quiet=False):
    # Returns Knudsen number and adiabaticity
    vT = np.sqrt(kB * temperature/mass)
    tc = 1.6e-4 * mass/(kB * temperature**8.0)
    mean_free_path = vT * tc
    characteristic_length = min(cell_dims)
    knudsen_number = mean_free_path/characteristic_length # <<1 : diffusive
    D = vT * vT * tc
    if knudsen_number < 1:
        # Diffusive regime
        adiabaticity = abs(w0) * characteristic_length**2 / D
    else:
        adiabaticity = abs(w0) * mean_free_path**2 / vT

    if not quiet:
        if abs(knudsen_number - 1) < 0.5:
            print('Warning: you might be in the transition regime (Knusden number close to 1)')

        print('Knudsen Number: {:.2e}'.format(knudsen_number))
        print('Adiabaticity: {:.2e}'.format(adiabaticity))
        print('w0 * tc: {:.2e}'.format(abs(w0) * tc))
        print('Regime: ' + ('diffusive ' if knudsen_number < 1 else 'ballistic ')
              + ('adiabatic' if adiabaticity > 1 else 'non-adiabatic'))

    return knudsen_number, adiabaticity

def phase_std_yx(y, x):
    return phase_std(np.arctan2(y, x))

def phase_std(ph):
    return np.std(np.mod(ph - ph[0] + np.pi, 2 * np.pi))

def analytic_spectrum(w, L=1, tau=1, vT=1, nq=10):
    total = 0
    for j in range(nq):#-nq, nq+1):
        n = (j - nq//2) * 2 + 1
        q = np.abs(n) * np.pi/L
        z = (1j * w + 1/tau)/(q * vT * np.sqrt(2))
        alpha = np.sqrt(np.pi/2)/(q * vT) * erfcx(z)
        total += n**(-4) * 1/(1/alpha - 1/tau)
    return 2 * 4 * L**2/np.pi**4 * total

class DressingBenchmark(Benchmark):
    # Allows you to compare to the Pignol result
    def __init__(self, physics=dict(), integration=dict(), executable=None, scratch_dir='.',
                 Gxx_range=[0], Gyy_range=[0], Gzz_range=[0],
                 Gxy_range=[0], Gyz_range=[0], Gzx_range=[0],
                 E_range=[0], T_range=[0.3], a_range=[0]):
        super().__init__(physics=physics, integration=integration, executable=executable, scratch_dir=scratch_dir)
        self.Gxx_range = Gxx_range # Array of dBx/dx values
        self.Gyy_range = Gyy_range # Array of dBy/dy values
        self.Gzz_range = Gzz_range # Array of dBz/dz values
        self.Gxy_range = Gyz_range # Array of dBx/dy values (dBy/dx constrained by Maxwell's)
        self.Gyz_range = Gyz_range # Array of dBy/dz values (dBz/dy constrained by Maxwell's)
        self.Gzx_range = Gzx_range # Array of dBz/dx values (dBx/dz constrained by Maxwell's)
        self.E_range = E_range # Array of Ex values
        self.T_range = T_range # Range of temperatures
        self.a_range = a_range # Range of dressing amplitudes
        self.output_files = []

    def run(self):
        field_iterator = itertools.product(self.Gxx_range, self.Gyy_range, self.Gzz_range,
                                          self.Gxy_range, self.Gyz_range, self.Gzx_range,
                                          self.E_range, self.T_range, self.a_range)
        for params in field_iterator:
            Gxx, Gyy, Gzz, Gxy, Gyz, Gzx, E, T, a  = params
            cell_dims = parse_triplet(self.config['L'])
            mass = float(self.config['m'])
            w0 = parse_triplet(self.config['B0'])[0] * float(self.config['gamma'])
            self.knudsen_number, self.adiabaticity = knudsen_and_adiabaticity(
                cell_dims, T, mass, w0
            )
            self.config['Gx'] =  '{:.2e}, {:.2e}, {:.2e}'.format(Gxx, Gxy, -Gzx)
            self.config['Gy'] = '{:.2e}, {:.2e}, {:.2e}'.format(-Gxy, Gyy, Gyz)
            self.config['Gz'] = '{:.2e}, {:.2e}, {:.2e}'.format(Gzx, -Gyz, Gzz)
            self.config['E'] = '{:.2e}, 0.0, 0.0'.format(E)
            self.config['T'] = '{:.2f}'.format(T)
            self.config['a'] = '{:.2e}'.format(a)
            out_file_name = 'out_{:d}.hdf5'.format(int(time.time() * 1000))
            self.output_files.append(os.path.join(self.scratch_dir, out_file_name))
            super().run(out_file_name)
            
    def analyze(self, files=None, time_indices=[-1]):
        if files is None:
            files = self.output_files
        col = Collector()
        for fname in files:
            with h5py.File(fname, 'r') as f:
                gamma = f.attrs['gamma']
                B0 = f.attrs['B0'][0]
                cell_dims = np.array([*f.attrs['L']])
                Gxx, Gxy, Gxz = f.attrs['Gx']
                Gyx, Gyy, Gyz = f.attrs['Gy']
                Gzx, Gzy, Gzz = f.attrs['Gz']
                E = f.attrs['E'][0]
                vT = f.attrs['sqrtKT_m']
                w = f.attrs['w']
                a = f.attrs['a']
                w0 = gamma * B0
                wprime = w0 * jv(0, gamma * a/w)
                G = np.array([[Gxx, Gxy, Gxz],
                              [Gyx, Gyy, Gyz],
                              [Gzx, Gyz, Gzz]])
                tc = f.attrs['tc']
                temperature = f.attrs['T']
                mass = f.attrs['m']
                self.knudsen_number, self.adiabaticity = knudsen_and_adiabaticity(
                    cell_dims, temperature, mass, w0, quiet=True
                )
                analytic_spectrum(w, L=cell_dims[2], tau=tc, vT=vT, nq=10)
                L2 = np.power(cell_dims, 2)
                bx2, by2, bz2 = np.power(G, 2) @ L2
                Sw0 = G[2,2]**2/4 * np.imag(analytic_spectrum(wprime, L=cell_dims[2], tau=tc, vT=vT))
                Swd1 = G[2,2]**2/4 * np.imag(analytic_spectrum(2 * w - wprime, L=cell_dims[2], tau=tc, vT=vT))
                Swd2 = G[2,2]**2/4 * np.imag(analytic_spectrum(2 * w + wprime, L=cell_dims[2], tau=tc, vT=vT))
                Sw0 += E**2/(c**4 * 2) * (1/2 * np.imag(analytic_spectrum(wprime, L=cell_dims[1], tau=tc, vT=vT))*wprime**2+ \
                                      1/2 * np.imag(analytic_spectrum(wprime, L=cell_dims[2], tau=tc, vT=vT))*wprime**2+ \
                                          wprime * (cell_dims[1]**2 + cell_dims[2]**2)/12) * jv(0, gamma * a/w)
                Swd1 = 0
                Swd2 = 0
                total_shift = -gamma**2 * (Sw0 + (jv(1, gamma * a/w) * w0/(1 * w))**2 * (Swd1 + Swd2))
                spin_data = np.stack((f['Spin']['x'], f['Spin']['y'], f['Spin']['z']))
                
                for time_idx in time_indices:
                    col.new_record()
                    b_end = np.mean(spin_data[:,time_idx,:], axis=1)

                    col.add('tf', f['Time'][time_idx][0])
                    phase_error = phase_std_yx(spin_data[1,time_idx,:], spin_data[2,time_idx,:])/np.sqrt(spin_data.shape[2])

                    col.add('Gxx', Gxx)
                    col.add('Gyy', Gyy)
                    col.add('Gzz', Gzz)
                    col.add('Gxy', Gxy)
                    col.add('Gyz', Gyz)
                    col.add('Gzx', Gzx)
                    col.add('E', E)
                    col.add('sx', b_end[0])
                    col.add('sy', b_end[1])
                    col.add('sz', b_end[2])
                    col.add('temperature', temperature)
                    col.add('tc', f.attrs['tc'])
                    col.add('ioutInt', f.attrs['ioutInt'])
                    col.add('phi', np.arctan2(b_end[1], b_end[2]))
                    col.add('phase_error', phase_error)
                    col.add('iType', f.attrs['integratorType'])
                    col.add('expected_shift', total_shift)

        data = col.to_pandas()
        phi0_data = data[(data['Gxx'] == 0) & (data['Gyy'] == 0) & (data['Gzz'] == 0) & \
                        (data['Gxy'] == 0) & (data['Gyz'] == 0) & (data['Gzx'] == 0) & (data['E'] == 0)]
        phi0 = np.array([phi0_data[(phi0_data['iType'] == it) & (phi0_data['tf'] == tf)]['phi'].iloc[0] for (it, tf) in zip(data['iType'], data['tf'])])
        return data.assign(shift = (np.mod((data['phi'] - phi0) + np.pi, 2 * np.pi) - np.pi)/data['tf'])
    
class PignolBenchmark(Benchmark):
    # Allows you to compare to the Pignol result
    def __init__(self, physics=dict(), integration=dict(), executable=None, scratch_dir='.',
                 Gxx_range=[0], Gyy_range=[0], Gzz_range=[0],
                 Gxy_range=[0], Gyz_range=[0], Gzx_range=[0],
                 E_range=[0], T_range=[0.3]):
        super().__init__(physics=physics, integration=integration, executable=executable, scratch_dir=scratch_dir)
        self.Gxx_range = Gxx_range # Array of dBx/dx values
        self.Gyy_range = Gyy_range # Array of dBy/dy values
        self.Gzz_range = Gzz_range # Array of dBz/dz values
        self.Gxy_range = Gyz_range # Array of dBx/dy values (dBy/dx constrained by Maxwell's)
        self.Gyz_range = Gyz_range # Array of dBy/dz values (dBz/dy constrained by Maxwell's)
        self.Gzx_range = Gzx_range # Array of dBz/dx values (dBx/dz constrained by Maxwell's)
        self.E_range = E_range # Array of Ex values
        self.T_range = T_range # Range of temperatures

        self.output_files = []

    def run(self):
        field_iterator = itertools.product(self.Gxx_range, self.Gyy_range, self.Gzz_range,
                                          self.Gxy_range, self.Gyz_range, self.Gzx_range,
                                          self.E_range, self.T_range)
        for params in field_iterator:
            Gxx, Gyy, Gzz, Gxy, Gyz, Gzx, E, T  = params
            cell_dims = parse_triplet(self.config['L'])
            mass = float(self.config['m'])
            w0 = parse_triplet(self.config['B0'])[0] * float(self.config['gamma'])
            self.knudsen_number, self.adiabaticity = knudsen_and_adiabaticity(
                cell_dims, T, mass, w0, quiet=True
            )
            self.config['Gx'] =  '{:.2e}, {:.2e}, {:.2e}'.format(Gxx, Gxy, -Gzx)
            self.config['Gy'] = '{:.2e}, {:.2e}, {:.2e}'.format(-Gxy, Gyy, Gyz)
            self.config['Gz'] = '{:.2e}, {:.2e}, {:.2e}'.format(Gzx, -Gyz, Gzz)
            self.config['E'] = '{:.2e}, 0.0, 0.0'.format(E)
            self.config['T'] = '{:.2f}'.format(T)
            out_file_name = 'out_{:d}.hdf5'.format(int(time.time() * 1000))
            self.output_files.append(os.path.join(self.scratch_dir, out_file_name))
            super().run(out_file_name)
            
    def analyze(self, files=None):
        if files is None:
            files = self.output_files
        col = Collector()
        for fname in files:
            with h5py.File(fname, 'r') as f:
                col.new_record()
                gamma = f.attrs['gamma']
                B0 = f.attrs['B0'][0]
                cell_dims = np.array([*f.attrs['L']])
                Gxx, Gxy, Gxz = f.attrs['Gx']
                Gyx, Gyy, Gyz = f.attrs['Gy']
                Gzx, Gzy, Gzz = f.attrs['Gz']
                E = f.attrs['E'][0]
                vT = f.attrs['sqrtKT_m']
                w0 = f.attrs['gamma'] * B0
                G = np.array([[Gxx, Gxy, Gxz],
                              [Gyx, Gyy, Gyz],
                              [Gzx, Gyz, Gzz]])
                tc = f.attrs['tc']
                
                temperature = f.attrs['T']
                mass = f.attrs['m']
                self.knudsen_number, self.adiabaticity = knudsen_and_adiabaticity(
                    cell_dims, temperature, mass, w0, quiet=True
                )
                L2 = np.power(cell_dims, 2)
                bx2, by2, bz2 = np.power(G, 2) @ L2
                if self.adiabaticity > 1:
                    expected_B2 = gamma**2/(2 * w0) * (by2 + bz2)/12 + \
                        gamma**2/(6 * w0**3) * 3 * vT**2 * (np.sum(G[1,:] * G[1,:]) + np.sum(G[2,:] * G[2,:]))
                    #expected_E2 = gamma**2 * E**2/(3 * c**4 * w0) * 3 * vT**2
                    expected_E2 = gamma**2 * E**2/(3 * c**4 * w0) * 3 * vT**2 * 1/(1/(w0 * tc)**2 + 1)
                    expected_BE = -gamma**2 * E/(c**2 * w0**2) * (Gyy * vT**2 + Gzz * vT**2)
                    #print(E, " ", expected_E2, " ", expected_BE, " ", expected_B2)
                else:
                    expected_B2 = 0 # Need an analytic expression for that integral
                    expected_E2 = -gamma**2 * E**2/(2 * c**4) * w0 * (L2[1] + L2[2])/12
                    expected_BE = gamma*2 * E/c**2 * (Gyy * L2[1] + Gzz * L2[2])/12
                
                total_shift = expected_B2 + expected_E2 + expected_BE
                    
                spin_data = np.stack((f['Spin']['x'], f['Spin']['y'], f['Spin']['z']))
                time_idx = -1                    
                b_end = np.mean(spin_data[:,time_idx,:], axis=1)
                
                phase_error = phase_std_yx(spin_data[1,-1,:], spin_data[2,-1,:])/np.sqrt(spin_data.shape[2])
                
                col.add('tf', f['Time'][-1][0])
                col.add('Gxx', Gxx)
                col.add('Gyy', Gyy)
                col.add('Gzz', Gzz)
                col.add('Gxy', Gxy)
                col.add('Gyz', Gyz)
                col.add('Gzx', Gzx)
                col.add('E', E)
                col.add('sx', b_end[0])
                col.add('sy', b_end[1])
                col.add('sz', b_end[2])
                col.add('temperature', temperature)
                col.add('tc', f.attrs['tc'])
                col.add('ioutInt', f.attrs['ioutInt'])
                col.add('phi', np.arctan2(b_end[1], b_end[2]))
                col.add('phase_error', phase_error)
                col.add('iType', f.attrs['integratorType'])
                col.add('expected_shift', total_shift)
                
        data = col.to_pandas()
        phi0_data = data[(data['Gxx'] == 0) & (data['Gyy'] == 0) & (data['Gzz'] == 0) & \
                        (data['Gxy'] == 0) & (data['Gyz'] == 0) & (data['Gzx'] == 0) & (data['E'] == 0)]
        if len(phi0_data) == 0:
            phi0_data = data[data['E'] == 0]
        phi0 = phi0_data['phi'].iloc[0]
        return data.assign(shift = (np.mod((data['phi'] - phi0) + np.pi, 2 * np.pi) - np.pi)/data['tf'])