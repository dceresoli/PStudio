#!/usr/bin/env python
from pstudio import AE, set_output
import numpy as np
import matplotlib.pyplot as plt
from math import pi

set_output('-')
#ae = AE('H', xcname='LDA', relativity='NR', config='1s1')
#ae = AE('Li', xcname='LDA', relativity='NR')
#ae = AE('Cl', xcname='LDA', relativity='SR')
#ae = AE('Ti', xcname='LDA', relativity='NR', config='[Ar] 3d2 4s1', rmin=1e-4)
ae = AE('Ti', xcname='LDA-py', relativity='SR')

try:
    ae.run(verbose=True)
except RuntimeError as err:
    print(err)

r = ae.rgd.r

# plot density
plt.figure()
plt.plot(r, ae.n*r*r, label='density')
plt.xlim(0,20)
plt.legend()
plt.show(block=False)

# plot the potential
plt.figure()
plt.plot(r, ae.vtot, label='Vtot')
plt.plot(r, ae.vion, label='Vion', linestyle='dashed')
plt.xlim(0,20)
plt.ylim(-10,0)
plt.legend()
plt.show(block=False)

plt.figure()
plt.plot(r, ae.vxc*r, label='Vxc*r')
plt.xlim(0,20)
plt.legend()
plt.show(block=False)

# plot orbitals
plt.figure()
for orb in ae.orbitals:
    plt.plot(r, orb.ur, label='n={0},l={1}'.format(orb.n,orb.l))
plt.xlim(0,10)
plt.legend()
plt.show()

quit()
