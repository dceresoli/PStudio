# PStudio - atomic and pseudopotentials calculations
# Copyright (C) 2020  Davide Ceresoli <dceresoli@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""Shooting methods for Boundary Value Problems"""

import numpy as np
from math import sqrt

# fine structure constant
alpha = 1/137.035999084


def shoot(u, h, c2, c1, c0):
    """
    Integrate differential equation by shooting method:

        u''(x)*c2(x) + u'(x)*c1(x) + u(x)*c0(x) = 0

    on an equispaced grid (grid spacing h). The DE is integrated with the
    Numerov method. Before calling, u[0:2] should be set in input according
    to boundary conditions. The ruotine retuns: u[], number of nodes,
    turning point and discontinuity of derivative at the turning point.
    """

    n = len(u)
    assert len(c0) == n
    assert len(c1) == n
    assert len(c2) == n

    # precalculate the Numerov recurrence terms
    fp = c2 + h*c1/2
    fm = c2 - h*c1/2
    f0 = (h*h)*c0 - 2*c2

    # inward integration up to turning point (or one point beyond to get derivative)
    # if no turning point, integrate half-way
    u[-1] = 1.0
    u[-2] = -u[-1]*f0[-1]/fm[-1]

    all_positive = np.all(c0 > 0.0)
    for i in range(n-2,0,-1):
        u[i-1] = (-fp[i]*u[i+1] - f0[i]*u[i]) / fm[i]
        if abs(u[i-1]) > 1e10:
            u[i-1:] *= 1e-10 # numerical stability
       	if c0[i] < 0.0:
            turn = i
            break
        if all_positive and i == n//2:
            turn = i
            break

    # derivative from the right
    uturn = u[turn]
    uturn1 = u[turn+1]
    dright = (u[turn+1] - u[turn-1]) / (2*h)

    # outward integration up to turning point (one point beyond to get derivative)
    for i in range(1,turn+1):
        u[i+1] = (-f0[i]*u[i] - fm[i]*u[i-1]) / fp[i]

    # derivative from the left
    dleft = (u[turn+1] - u[turn-1]) / (2*h)

    # rescale u in order to make it continuos
    scale = uturn / u[turn]
    u[:turn+1] *= scale
    u[turn+1] = uturn1
    dleft *= scale

    # set the sign
    u = u*np.sign(u[1])

    # count nodes
    nodes = np.sum( (u[0:turn-1]*u[1:turn]) < 0.0 )
    derdisc = (dright - dleft) * uturn

    return u, nodes, turn, derdisc


def setup_coefficients(rgd, l, v, ene, srel=False):
    """Setup the c0(x), c1(x) and c2(x) functions"""

    # add the centrigucal term
    N = rgd.npoints
    r = rgd.r
    #vpot = v + l*(l+1)/(2.0*r*r)

    if srel == False:
        # non relativistic case
        c0 = 2.0*r*r*(v - ene) + l*(l+1)
        c1 = np.ones(N)
        c2 = -np.ones(N)
    else:
        # scalar relativistic
        mass = 1 - alpha*alpha/2.0 * (v - ene)
        dvdr = rgd.fdiff_deriv1(v)
        kappa = -1

        tmp = 0.5*alpha*alpha*dvdr/mass
        c0 = 2.0*r*r*mass*(v-ene) + l*(l+1) - tmp*kappa*r
        c1 = 1.0 - tmp*r
        c2 = -np.ones(N)

    return c0, c1, c2


def solve_rsched(rgd, n, l, v, srel=False, debug=False):
    """
    Solve the radial Schroedinger equation (non relativistic and scalar
    relativistic case) for a given n,l and return the radial wavefunction
    and its eigenvalue
    """
    assert 0 <= l < n

    # setup potential
    N = rgd.npoints
    zeta = rgd.zeta
    r = rgd.r
    h = rgd.dx

    # initila guess
    emin, emax = -1.1*zeta*zeta/(n*n), 0.0
    if l == 0:
        emin = np.min(v + 0.25/r**2)
    else:
        emin = np.min(v + 0.5*l*(l+1)/r**2)
    ene = (emin + emax)/2.0

    expected_nodes = n - l - 1

    # start iteration
    iteration = 0
    maxiter = 100
    while True:
        iteration += 1

        # setup coefficients
        c0, c1, c2 = setup_coefficients(rgd, l, v, ene, srel)

        # setup boundary conditions
        if srel:
            if l == 0:
                gamma = sqrt(1 - alpha*alpha*Z*Z)
            else:
                gamma = ( l*sqrt(l*l-alpha*alpha*Z*Z) + (l+1)*sqrt((l+1)*(l+1)-alpha*alpha*Z*Z) ) / (2*l+1)
        else:
            gamma = l + 1
        u = np.zeros(N)
        u[0:2] = r[0:2]**gamma

        # integrate
        u, nodes, turn, derdisc = shoot(u, h, c2, c1, c0)
        norm = rgd.integrate(u*u)
        u /= sqrt(norm)
        if debug:
            print("Iteration %3i:   Energy = %12.6f   Nodes = %i (%i)" % (iteration, ene, nodes, expected_nodes))

        if iteration > maxiter:
            if debug:
                print("wavefunction (%i,%i) not converged" % (n,l))
            ene = 0
            return u, ene

        if nodes == expected_nodes:
            # refine energy
            shift = -0.5 * derdisc / (r[turn]*norm)
            if shift > 0.0:
                emin = ene
            else:
                emax = ene

            ene += shift
            if ene < emin or ene > emax:
                ene = (emin + emax)/2.0

            if abs(shift) < 1e-10:
                return u, ene

        elif nodes > expected_nodes:
            emax = ene
            ene = (emin + emax)/2.0
            continue

        elif nodes < expected_nodes:
            emin = ene
            ene = (emin + emax)/2.0
            continue


if __name__ == '__main__':
    import sys
    sys.path.append('..')

    from pstudio.radialgrid import RadialGrid
    from pstudio.util import thomas_fermi
    from pstudio.oncvpsp_routines.oncvpsp import lschfb
    import matplotlib.pyplot as plt

    Z = 92
    n, l = 1, 0
    srel = True
    rgd = RadialGrid(zeta = Z, rmin=1e-5)

    print('HYDROGENOIC -------------')
    u, ene =solve_rsched(rgd, n, l, -Z/rgd.r, srel=srel, debug=True)
    et = np.array([0.0])
    ierr, ur, _, match = lschfb(n, l, et, rgd.r, -Z/rgd.r, Z, srel)
    print('lschfb: e={0}, ierr={1}'.format(et[0], ierr))
    print('error={0:.4f} ({1:.4f}%)'.format(abs(ene-et[0]), 100*(ene-et[0])/et[0]))
    if not srel:
        print('exact NR energy:',-0.5*Z*Z/(n*n))
    print()

    print('THOMAS-FERMI ------------')
    u2, ene2 = solve_rsched(rgd, n, l, thomas_fermi(Z, rgd.r), srel=srel, debug=True)
    et = np.array([0.0])
    ierr, ur, _, match = lschfb(n, l, et, rgd.r, thomas_fermi(Z, rgd.r), Z, srel)
    print('lschfb: e={0}, ierr={1}'.format(et[0], ierr))
    print('error={0:.4f} ({1:.4f}%)'.format(abs(ene2-et[0]), 100*(ene2-et[0])/et[0]))
    quit()

    r = rgd.r
    plt.figure(dpi=200)
    plt.plot(r, -Z/r, label='Coulomb')
    plt.plot(r, thomas_fermi(Z, r), label='Thomas-Fermi')
    plt.xlim(0,5)
    plt.ylim(-100,0)
    plt.title('Z={0}'.format(Z))
    plt.legend()
    plt.show(block=False)

    plt.figure(dpi=200)
    plt.plot(r, u, label='Coulomb')
    plt.plot(r, u2, label='Thomas-Fermi')
    plt.xlim(0,5)
    plt.title('Z={0}, n={1}, l={2}'.format(Z, n, l))
    plt.legend()
    plt.show(block=False)

    plt.show()
