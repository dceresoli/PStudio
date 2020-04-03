#!/usr/bin/env python
from pstudio import AE
import numpy as np
import matplotlib.pyplot as plt
from math import pi

#ae = AE('H', xcname='LDA', relativity='NR', config='1s1', out='-')
#ae = AE('Li', xcname='LDA', relativity='NR', out='-')
ae = AE('Ti', xcname='LDA', relativity='SR', out='-', rmin=0.1524e-4, npoints=1963)
#ae = AE('Cl', xcname='LDA', relativity='SR', out='-')

try:
    ae.run(verbose=True)
except RuntimeError as err:
    print(err)

r = ae.rgd.r
with open('r.dat', 'w') as f:
    np.savetxt(f, r)

from scipy.interpolate import InterpolatedUnivariateSpline
import sys
ld1_rho = np.loadtxt('Rho.dat')
interp = InterpolatedUnivariateSpline(ld1_rho[:,0],ld1_rho[:,1])
ae.n = interp(r) / (4.0*pi)/(r*r)
ae.calculate_potential()
ae.calculate_energies()
print(ae.Etot)

# plot density
#plt.plot(r, ae.n*r*r, label='density')
#plt.plot(ld1_rho[:,0], ld1_rho[:,1] /(4.0*pi), label='LD1 density')
plt.plot(r, ae.n*r*r - interp(r)/(4.0*pi), label='density')
plt.axhline(0, color='k', linewidth=0.5)
plt.xlim(0,20)
plt.legend()
plt.show()

# plot the potential
ld1_vxc = np.loadtxt('VHXC.dat')
#plt.plot(r, ae.vtot*r, label='Vtot*r')
#plt.plot(r, ae.vion*r, label='Vion*r', linestyle='dashed')

#plt.plot(r, ae.vxc*r, label='Vxc*r')
#plt.plot(ld1_vxc[:,0], ld1_vxc[:,2]*ld1_vxc[:,0]/2, label='LD1 Vxc*r')
plt.plot(r, ae.vh*r, label='VH')
plt.plot(ld1_vxc[:,0], ld1_vxc[:,1]/2*ld1_vxc[:,0], label='LD1 VH')
plt.xlim(0,20)
#plt.ylim(21,23)
plt.legend()
plt.show()

# plot orbitals
ld1 = np.loadtxt('ti.wfc')
for orb in ae.orbitals:
    if orb.n == 4 and orb.l == 0 and True:
        plt.plot(r, orb.ur, label='n={0},l={1}'.format(orb.n,orb.l))
        plt.plot(ld1[:,0], -ld1[:,1], label='LD1 4s')

    #if orb.n == 1 and orb.l == 0:
    #    plt.plot(r, orb.ur, label='n={0},l={1}'.format(orb.n,orb.l))
    #    plt.plot(ld1[:,0], ld1[:,-1], label='LD1 1s')
plt.xlim(0,10)
plt.legend()
plt.show()

quit()
