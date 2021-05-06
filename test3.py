#!/usr/bin/env python
from pstudio import AE, set_output
from pstudio.all_electron import AEwfc
from pstudio.periodic_table import tuple_to_configuration
from pstudio.TM import pseudize_TM
from pstudio.RRKJ import pseudize_RRKJ
from pstudio.vloc import generate_vloc_RRKJ, generate_vloc_TM
from pstudio.pseudo import  calculate_vpot
from pstudio.confinement import ConfinementPotential
import numpy as np
import matplotlib.pyplot as plt
from math import pi

set_output('-')
r0 = 3.5/0.52917
confinement = ConfinementPotential('woods-saxon', r0=r0, W=2.0, a=4.0)
#ae = AE('Si', config='[Ne] 3s2 3p2 3d0 4s0 4p0 4d0 4f0 5s0 5p0 5d0 6s0 6p0', xcname='LDA', relativity='SR', confinement=confinement)
ae = AE('Pt', config='[Xe] 4f14 5d9 6s1 6p0 7s0 7p0 6d0 5f0', xcname='LDA-py', relativity='SR', confinement=confinement)

try:
    ae.run(verbose=True)
except RuntimeError as err:
    print(err)

#orb = AEwfc(len(ae.rgd), 3, 2, 0.0)
#orb.ur, orb.e, ierr = ae.solve_orbital(3, 2)
#print(ierr)
#ae.orbitals.append(orb)
#print(orb.e)

# plot orbitals
r = ae.rgd.r
for orb in ae.orbitals:
    conf = tuple_to_configuration([(orb.n, orb.l, orb.f)])
    plt.plot(r, orb.ur, label='AE '+conf)
    plt.xlim(0,10)
    plt.legend()
plt.show()

with open('si-aepot.dat', 'w') as f:
    for i in range(len(ae.rgd)):
        f.write('{0} {1}\n'.format(ae.rgd.r[i], ae.vtot[i]))

#with open('si-aeorb.dat', 'w') as f:
#    f.write('{0}\n'.format(len(ae.rgd)))
#    for orb in ae.orbitals:
#        f.write('{0} {1}\n'.format(orb.n, orb.l))
#        for i in range(len(ae.rgd)):
#            f.write('{0:20.12f} {1:20.12f}\n'.format(ae.rgd.r[i], orb.ur[i]))

for orb in ae.orbitals:
    label = '{0}{1}.dat'.format(orb.n,'SPDF'[orb.l])
    with open(label, 'wt') as f:
        np.savetxt(f, np.column_stack((ae.rgd.r, orb.ur)))

quit()




ae.calculate_tau()
orb = ae.orbitals[1]
l = orb.l
rmax = orb.find_rmax(ae.rgd)
aeorb = orb.ur
psorb, d2psorb = pseudize_RRKJ(aeorb, l, rmax*1.2, ae.rgd, verbose=True)
psorb, d2psorb = pseudize_TM(aeorb, l, rmax*1.2, ae.rgd, verbose=True)
vpot = calculate_vpot(ae.vtot, ae.rgd, rmax*1.2, orb.l, orb.e, psorb, d2psorb)

generate_vloc_RRKJ(ae.vtot, ae.rgd, 1.2, verbose=True)
generate_vloc_TM(ae.vtot, ae.rgd, 1.2, verbose=True)

orb = ae.orbitals[1]

# plot orbitals
r = ae.rgd.r
plt.figure()
plt.plot(r, aeorb, label='AE')
plt.plot(r, psorb, label='PS')
plt.xlim(0,10)
plt.legend()
plt.show()

quit()
