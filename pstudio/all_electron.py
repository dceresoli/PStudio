# PStudio - atomic and pseudopotentials calculations
# Copyright (C) 2010  Davide Ceresoli <dceresoli@gmail.com>
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
"""Atomic all electron calculation"""

import sys, os
import pickle

from math import pi, sqrt, log
import numpy as np
from .xc import XC

from .configuration import *
from .radialgrid import RadialGrid
from .oncvpsp_routines.oncvpsp import lschfb

hartree = 27.211386

class AEwfc:
    # TODO: add normalize, initialize, plot, save, etc...
    """All-electron atomic orbital"""
    def __init__(self, npoints, n , l, f, e=0.0, spin=1):
        assert n >= 0
        assert l < n
        assert 1 <= spin <= 2
        self.n = n
        self.l = l
        self.f = f
        self.e = e
        self.spin = spin
        self.ur = np.zeros(npoints)   # wavefunction
        self.up = np.zeros(npoints)   # derivative

    def normalize(self, rgd):
        """Normalize a wavefunction"""
        norm = rgd.integrate(self.ur*self.ur)
        self.ur /= sqrt(norm)

    def find_rmax(self, rgd):
        """Find outermost maximum"""
        imax = np.argmax(np.abs(self.ur))
        rmax = rgd.r[imax]
        return rmax

    def make_positive(self, rgd, rmax):
        """Make wfc positive at the outermost radius"""
        imax = rgd.ceil(rmax)
        if self.ur[imax] < 0.0:
            self.ur = -self.ur

class frozen:
    __isfrozen = False
    def __setattr__(self, key, value):
        if self.__isfrozen and not hasattr(self, key):
            raise TypeError( "%r is a frozen class" % self )
        object.__setattr__(self, key, value)

    def _freeze(self):
        self.__isfrozen = True


class AE(frozen):
    """Object for doing an atomic DFT calculation."""

    def __init__(self, symbol, xcname='LDA', relativity='SR', config=None,
                 out='-', rmin=1e-5, rmax=100, npoints=2001):
        """Perform an AE atomic DFT calculation.

        Example::

          fe = AE('Fe', xcname='PBE')
          fe.run(restart=False)
        """

        # initialize output
        if out == '-':
            out = sys.stdout
        elif isinstance(out, str):
            out = open(out, 'w')
        self.out = out

        # initialize variables
        self.symbol = symbol
        self.xc = XC(xcname)
        self.relativity = relativity

        # initialize radial grid
        el = Element(symbol)
        self.Z = el.get_atomic_number()
        self.N = npoints
        self.rgd = RadialGrid(self.Z, rmin, rmax, self.N)

        # initialize configuration
        if config is None:
            nlf = parse_configuration(el.get_configuration())
        else:
            nlf = parse_configuration(config)
        self.orbitals = []
        self.nelec = 0
        for n, l, f in nlf:
            ene = -0.5 * self.Z**2 / n**2
            self.nelec += f
            self.orbitals.append(AEwfc(npoints, n, l, f, ene))

        # density, potential and energies
        self.n = np.zeros(npoints)
        self.vh = np.zeros(npoints)
        self.vxc = np.zeros(npoints)
        self.vion = -self.Z / self.rgd.r
        self.vtot = np.zeros(npoints)
        self.Etot = 0.0        # total energy
        self.Ekin = 0.0        # kinetic energy
        self.Eion = 0.0        # ionic energy
        self.Eh = 0.0          # Hartree energy
        self.Exc = 0.0         # XC energy
        self.Evxc = 0.0        # integral of rho*vxc
        self.restartfile = '%s-AE.npz' % (self.symbol)
        self._freeze()         # no more attributes

        # print summary
        p = self._print
        p()
        try:
            rel = {'NR': 'non relativistic', 'SR': 'scalar relativistic',
                   'FR': 'fully relativistic'}[relativity]
        except KeyError:
            raise RuntimeError('Unknown relavistic method (SR, NR and FR accepted)')
        p('{0} atomic calculation for {1} ({2}, Z={3})'.format(rel, symbol, el.get_name(), self.Z))
        p('configuration: {0}, {1:g} electrons'.format(tuple_to_configuration(nlf),self.nelec))
        p('exchange-correlation: {0}'.format(self.xc.get_name()))

    def _print(self, *args, **kwargs):
        """Helper routine to output data to a file or to stdout"""
        if self.out is None:
            return
        print(*args, **kwargs, file=self.out)


    def initialize_wave_functions(self):
        """Initialize with hydrogenoic wave functions"""
        for orb in self.orbitals:
            orb.ur[:] = hydrogen_wfc(self.rgd.r, self.Z, orb.n, orb.l)
            orb.normalize(self.rgd)


    def run(self, restart=False, verbose=False, mixing=0.2, thresh=1e-6):
        """Perform an all-electron SCF calculation"""
        p = self._print

        N = self.N
        r = self.rgd.r
        dr = self.rgd.dr
        Z = self.Z
        p('{0} radial gridpoints in [{1:g},{2:g}]'.format(N, r[0], r[-1]))

        # try to restart the density
        if restart:
            try:
                self.n = np.load(restartfile, 'r')['rho']
            except:
                restart = False
            else:
                self.calculate_potential()

        # from scratch or if restart was not succesful
        if not restart:
            self.vtot = self.vion.copy()

        # SCF cycle
        nitermax = 300
        n_old = np.zeros(N)
        Etot_old = 0.0

        for niter in range(1, nitermax):
            # solve radial Schrodinger equation for each orbital
            self.solve()

            # rho mixing
            charge = self.calculate_density()
            if niter > 1:
                self.n = mixing * self.n + (1.0-mixing) * n_old
            n_old = self.n.copy()

            # calculate hartree and XC potential
            self.calculate_potential()
            self.vtot = self.vion + self.vh + self.vxc

            # calculate energy terms
            self.calculate_energies()
            de = self.Etot - Etot_old
            Etot_old = self.Etot
            if verbose == True:
                if niter == 1:
                    p()
                    p('iteration      energy          delta        charge')
                    p('-'*72)
                p('{0:3d}      {1:12.6f}   {2:12.4e}  {3:12.6f}'.format(niter, self.Etot, de, charge))

            if abs(de) < thresh:
                break

            if niter >= nitermax:
                raise RuntimeError('Too many iterations!')

        p()
        p('Converged in {0} iterations'.format(niter))

        self.calculate_density()
        self.calculate_potential()
        self.print_energies()
        self.print_eigenvalues()

        #self.write_restart()
        # TODO: fix this
        # write restart file
        #try:
        #    fd = open(restartfile, 'wb')
        #except IOError:
        #    pass
        #else:
        #    pickle.dump(n, fd)
        #    try:
        #        os.chmod(restartfile, 0o666)
        #    except OSError:
        #        pass
        # TODO save to npz file
        #for m, l, u in zip(n_j, l_j, self.u_j):
        #    self.write(u, 'ae', n=m, l=l)
        #
        #tau = self.calculate_kinetic_energy_density()
        #self.write(n, 'n')
        #self.write(vr, 'vr')
        #self.write(vHr, 'vHr')
        #self.write(self.vXC, 'vXC')
        #self.write(tau, 'tau')

    def solve(self):
        """Solve the radial Schrodinger equation using oncvpsp routines"""
        if self.relativity == 'FR':
            raise NotImplementedError('FR not implemented yet!')

        # in inpurt vr is the potential*r
        srel = (self.relativity == 'SR')

        # solve for each quantum state separately
        for orb in self.orbitals:
            n, l, f, e = orb.n, orb.l, orb.f, orb.e
            if f < 0.0:  # do not solve unbound states (negative occupations)
                orb.ur = np.zeros_like(self.rgd.r)
                orb.up = np.zeros_like(self.rgd.r)
                orb.e = 0.0
                continue

            et = np.array(e)
            ierr, orb.ur, orb.up, match = lschfb(n, l, et, self.rgd.r, self.vtot, self.Z, srel)
            if ierr == 0:
                orb.e = et
                orb.normalize(self.rgd)
            else:
                orbital = tuple_to_configuration([(n,l,f)])
                orb.ur[:] = 0.0
                #if f >= 0.0:
                #    raise Warning('Error solving orbital {0}'.format(orbital))
                #else:
                #    raise Warning('Problem solving orbital {0}'.format(orbital))


    def calculate_density(self):
        """Calculate the electron charge density divided by 4pi"""
        self.n = np.zeros(self.N)
        for orb in self.orbitals:
            self.n += orb.f * orb.ur*orb.ur
        charge = self.rgd.integrate(self.n)
        self.n = self.n / (4.0*pi * self.rgd.r**2)
        return charge


    def calculate_energies(self):
        self.Ekin = 0.0
        for orb in self.orbitals:
            self.Ekin += orb.f * orb.e
        self.Etot = self.Ekin - self.Eh + self.Exc - self.Evxc
        self.Ekin = self.Etot - self.Eion - self.Eh - self.Exc

    def calculate_potential(self):
        self.Eh, self.vh = self.calculate_hartree(self.n)
        self.Eion = self.rgd.integrate(self.vion * self.n*self.rgd.r**2) * 4.0*pi
        self.calculate_xc(self.n)
        self.Exc, self.vxc, self.Evxc = self.calculate_xc(self.n)
        self.vtot = self.vh + self.vxc + self.vion

    def calculate_hartree(self, rho):
        """Calculate the Hartree potential from a given density"""
        v_h = self.rgd.hartree(rho*4*pi)
        e_h = 0.5*self.rgd.integrate(v_h * rho * self.rgd.r**2) * 4.0*pi
        return e_h, v_h

    def calculate_xc(self, rho):
        """Calculate the XC potential from a given density"""
        r = self.rgd.r

        e_xc, v_xc = self.xc.compute(rho, None)

        e_xc = self.rgd.integrate(e_xc * rho * r**2) * 4.0*pi
        e_vxc = self.rgd.integrate(v_xc * rho * r**2) * 4.0*pi
        return e_xc, v_xc, e_vxc

    def print_energies(self):
        """Print energy terms"""
        p = self._print
        p()
        p('Energy contributions:')
        p('-'*72)
        p('Kinetic:   {0:+13.6f} Ha    {1:+13.6f} eV '.format(self.Ekin,self.Ekin*hartree))
        p('Ionic:     {0:+13.6f} Ha    {1:+13.6f} eV '.format(self.Eion,self.Eion*hartree))
        p('Hartree:   {0:+13.6f} Ha    {1:+13.6f} eV '.format(self.Eh,self.Eh*hartree))
        p('XC:        {0:+13.6f} Ha    {1:+13.6f} eV '.format(self.Exc,self.Exc*hartree))
        p('-'*72)
        p('Total:     {0:+13.6f} Ha    {1:+13.6f} eV '.format(self.Etot,self.Etot*hartree))


    def print_eigenvalues(self):
        p = self._print

        p()
        p('state      eigenvalue      eigenvalue        rmax')
        p('-'*72)
        for orb in self.orbitals:
            label = tuple_to_configuration([(orb.n,orb.l,orb.f)])
            rmax = orb.find_rmax(self.rgd)
            orb.make_positive(self.rgd, rmax)
            p('{0:8s} {1:12.6f} Ha {2:12.6f} eV {3:8.3f}'.format(label, \
             orb.e, orb.e*hartree, rmax))
        p()



#    def calculate_kinetic_energy_density(self):
#        """Return the kinetic energy density"""
#        return self.radial_kinetic_energy_density(self.f_j, self.l_j, self.u_j)
#
#    def radial_kinetic_energy_density(self, f_j, l_j, u_j):
#        """Kinetic energy density from a restricted set of wf's"""
#        shape = np.shape(u_j[0])
#        dudr = np.zeros(shape)
#        tau = np.zeros(shape)
#        for f, l, u in zip(f_j, l_j, u_j):
#            self.rgd.derivative(u, dudr)
#            # contribution from angular derivatives
#            if l > 0:
#                tau += f * l * (l + 1) * np.where(abs(u) < 1e-160, 0, u)**2
#            # contribution from radial derivatives
#            dudr = u - self.r * dudr
#            tau += f * np.where(abs(dudr) < 1e-160, 0, dudr)**2
#        tau[1:] /= self.r[1:]**4
#        tau[0] = tau[1]
#
#        return 0.5 * tau / (4 * pi)



if __name__ == '__main__':
    a = AEAtom('Ti')
    a.run()
