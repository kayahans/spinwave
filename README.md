# spinwave
Collinear spinwave calculations

Calculates linear spinwave theory Hamiltonians

This is the collinear only version of the work in J. Phys.: Condens. Matter 27 (2015) 166002.

For 2D materials only, this code can also calculate the magnetization vs. temperature curves. 

If you use this code, please cite:
https://arxiv.org/abs/2205.00300 and
https://iopscience.iop.org/article/10.1088/0953-8984/27/16/166002/meta


Here is an example given for the ferromagnetic CrI3 using magnetic coupling parameters in 2D Mater. 4 (2017) 035002.
```python
import matplotlib.pyplot as plt

sw = Spinw()
# POSCAR should be in the "directory as below"
# If OUTCAR is also in the same directory, magnetic moments are read from OUTCAR.
# If not, then each atomic magnetic moment must be supplied in "spins".
sw.read_structure(directory="./", spins=[3,3,0,0,0,0,0,0])
  
# define magnetic interactions
# This example uses 2D CrI3 structure. 
# Reference values are taken from 2D Mater. 4 (2017) 035002:
#   * isotropic exchange J is defined as 2.2 meV
#   * anisotropic exchange lambda is defined as 0.09 meV
# Negative sign of J denotes that interactions are ferromagnetic.
# In XXZ hamiltonian, lambda is the anisotropic exchange in z-direction
# Hence it is the 3rd element of the np.diag([0,0,-0.09]) vector below.
# For lambda, the convention is to use negative sign for the favored 
# anisotropy direction and use zero for the unfavorable directions.
# rmax = 4.5 A means only first nearest neighbors for this example

sw.add_coupling(('Cr1', 'Cr2'), -2.2, rmax=4.5)
sw.add_coupling(('Cr1', 'Cr2'), np.diag([0,0,-0.09]), rmax=4.5)
    
# plot magnon band structure
# compare to 2D Mater. 4 (2017) 035002
ax = plt.figure().gca()
sw.get_band_structure(ax, [0.0,0.0,0.0], [0.333333, 0.3333333, 0.0], num_k=100, normalize = True)
# normalize tag normalizes the energy axis in the band structure to JS units. 
figname = './band_line.pdf'
plt.xlabel('k*l')
plt.savefig(figname, format='pdf')

# plot magnetization curve 
sw.magnetization(kgrid=[20,20,1], tmin = 1, tmax = 100, dt = 1)
```

CrI3 magnon dispersion curve:

<img src="/examples/cri3/band_line.png" alt="band" width="500"/>

CrI3 magnetization curve:

<img src="/examples/cri3/CrI3_magnetization.png" alt="magnet" width="500"/>

