#!/usr/bin/env python
from pstudio import AE, set_output
from pstudio.all_electron import AEwfc
from pstudio.periodic_table import tuple_to_configuration, parse_configuration, Element
from pstudio.confinement import ConfinementPotential
import numpy as np
from math import pi
import os, os.path

set_output('-')

r0 = 4.0/0.52917
confinement = ConfinementPotential('woods-saxon', r0=r0, W=2.0, a=4.0)

try:
    os.mkdir('BASIS')
except FileExistsError:
    pass

for z in range(1,95):
    el = Element(z)
    print(el.name, el.configuration)

    config = parse_configuration(el.configuration)
    n_max = [0,1,2,3]
    for (n,l,occ) in config:
        if n > n_max[l]:
            n_max[l] = n

    for l in (0,1,2,3):
        config.append((n_max[l]+1,l,0.0))

    print(el.name, tuple_to_configuration(config))

    try:
        os.mkdir('BASIS/'+el.symbol)
    except FileExistsError:
        pass

    ae = AE(el.symbol, config=tuple_to_configuration(config), xcname='LDA', relativity='SR', confinement=confinement)

    try:
        ae.run(verbose=True)
    except RuntimeError as err:
        print(err)

    for orb in ae.orbitals:
        label = '{0}{1}.dat'.format(orb.n,'SPDF'[orb.l])
        with open(os.path.join('BASIS', el.symbol, label), 'wt') as f:
            np.savetxt(f, np.column_stack((ae.rgd.r, orb.ur)))

quit()


