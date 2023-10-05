import numpy as np
import re
from matplotlib import pyplot
import qutip

B0 = 3e-6
B1 = 3e-5
he3gyro = -2.038e8

delta = he3gyro * B0
A     = he3gyro * B1
omega = 1.0e3 * 2*np.pi
T      = (2*np.pi)/omega
w_th = 0

def J_cb(w):
    return 1/((w*T)**2 + 1)

H0 = delta/2.0 * qutip.sigmax()
H1 = A/2.0 * qutip.sigmaz()
args = {'w': omega}
H = [H0, [H1, lambda t,args: np.cos(args['w'] * t)]]
c_op = qutip.sigmay()

n_prop = 100
tlist = np.linspace(0, T, n_prop + 1)
# find the floquet modes for the time-dependent hamiltonian
options = qutip.Options(atol=1e-14, rtol=1e-14)
f_modes_0,f_energies = qutip.floquet_modes(H, T, args, options=options)
f_modes_table_t = qutip.floquet_modes_table(f_modes_0, f_energies, tlist, H, T, args, options=options)
Delta, X, Gamma, A = qutip.floquet_master_equation_rates(f_modes_0, f_energies, c_op, H, T, args, J_cb, w_th, kmax=5, f_modes_table_t=f_modes_table_t)

rho0 = (qutip.sigmay() + qutip.qeye(2))/2
R = qutip.floquet_master_equation_tensor([A], f_energies)
res = qutip.floquet_markov_mesolve(R, rho0, [0, T * 10], [qutip.sigmax(), qutip.sigmay(), qutip.sigmaz()], options=options).expect

bx, by, bz = res
b = np.array([bx[1], by[1], bz[1]])

def save(fname, array):
    s = np.array_str(array)
    s = s.replace('\n', '').replace('\r', '')
    s = re.sub(r' +\]', ']', re.sub(r'\[ +', '[', s))
    s = re.sub(r' +', ', ', s).replace('[', '{').replace(']', '}') + '\n'
    with open(fname, 'w') as f:
        f.write(s)

import sys
import os
from os.path import join
scripts_directory = os.path.dirname(os.path.abspath(sys.argv[0]))
data_directory = 'floquet_matrices'
directory = join(scripts_directory, 'floquet_matrices')
print(f'Saved tensors to {directory}')
save(join(directory, 'Delta.txt'), Delta)
save(join(directory, 'X.txt'), np.abs(X)*np.abs(X))
save(join(directory, 'Gamma.txt'), Gamma)
save(join(directory, 'A.txt'), A)
save(join(directory, 'b.txt'), b)
