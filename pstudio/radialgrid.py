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
"""Logarithmic grid for atomic calculations"""

from math import log, pi
from math import factorial as fac
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from .oncvpsp_routines.oncvpsp import hartree as _hartree

class RadialGrid:
    def __init__(self, zeta=1.0, rmin=0.0001, rmax=100.0, npoints=2001):
        """
        Initialize a radial logarithmic grid (exp(x)/Z), for nuclear charge zeta
        """
        # store parameters
        self.zeta = zeta
        self.npoints = npoints
        # construct logaritmic mesh
        self.rmin = rmin
        self.rmax = rmax
        self.xmin, self.xmax = log(zeta*rmin), log(zeta*rmax)
        self.x =  np.linspace(self.xmin, self.xmax, npoints)
        self.dx = (self.x[-1] - self.x[0])/npoints
        self.dx = self.x[1] - self.x[0]
        self.r = np.exp(self.x)/zeta
        self.dr = self.dx * self.r

    def __len__(self):
        return self.npoints

    def r_of_x(self, x):
        return np.exp(x)/self.zeta

    def r2g(self, r):
        x = np.log(self.zeta*r)
        return (x-self.xmin)/self.dx

    def floor(self, r):
        return np.floor(self.r2g(r)).astype(int)

    def round(self, r):
        return np.around(self.r2g(r)).astype(int)

    def ceil(self, r):
        return np.ceil(self.r2g(r)).astype(int)

    def integrate(self, f):
        """Integrate a function using the trapezium rule"""
        res = f[0]*self.dr[0] + f[-1]*self.dr[-1]
        res += 2.0*np.sum(f[1:-1]*self.dr[1:-1])
        return res/2.0

    def interpolate(self, f, r_x):
        """Interpolate a function on a new grid"""
        return InterpolatedUnivariateSpline(self.r, f)(r_x)

    def hartree(self, rho):
        """Return the Hartee potential given a density"""
        return _hartree(rho, self.zeta, self.r)

    #def deriv1(self, f):
    #    """Calculate the 1st derivative of a function"""
    #    return fdiff1_7p(f)/self.dx / self.r

    #def deriv2(self, f):
    #    """Calculate the 1st derivative of a function"""
    #    return (fdiff2_7p(f)/self.dx**2 - fdiff1_7p(f)/self.dx) / self.r**2

    def deriv1(self, f):
        """Calculate the 1st derivative of a function"""
        r = self._decimate_towards_zero(self.r)
        g = self._decimate_towards_zero(f)
        spl = InterpolatedUnivariateSpline(r, g, k=3)
        return spl.derivative(n=1)(self.r)

    def deriv2(self, f):
        """Calculate the 1st derivative of a function"""
        r = self._decimate_towards_zero(self.r)
        g = self._decimate_towards_zero(f)
        spl = InterpolatedUnivariateSpline(r, g, k=3)
        return spl.derivative(n=2)(self.r)

    def _decimate_towards_zero(self, func):
        m = self.floor(0.01)
        return np.concatenate( (func[:m:20], func[m:]) )

    def kinetic(self, f, l):
        """Apply the kinetic energy of a function"""
        return -0.5*self.deriv2(f) + 0.5*(l*(l+1))/self.r**2

    def fft(self, f, l=0, N=None):
        """Fourier transform, return |G| and f(|G|) arrays"""
        if N is None:
            N = 2**13

        assert N % 2 == 0
        # interpolate on a uniform fine grid
        r_x = np.linspace(0.0, self.rmax, N)
        f_x = self.interpolate(f, r_x)
        f_x[1:] /= r_x[1:]
        f_x[0] = f_x[1]
        G_k = np.linspace(0, pi/r_x[1], N//2+1)
        f_k = 4 * pi * fsbt(l, f_x, r_x, G_k)   # fast spherical bessel
        return G_k, f_k


# utility funcionts

def fdiff1_7p(f):
    """7-points 1st finite differences"""
    n = len(f)
    df = np.zeros_like(f)

    forw_coeff = np.array([-49/20, 6, -15/2, 20/3, -15/4, 6/5, -1/6])
    for i in range(3):
        df[i] = np.sum(forw_coeff * f[i:i+7])

    cent_coeff = np.array([-1/60, 3/20, -3/4, 0, 3/4, -3/20, 1/60])
    for i in range(3,n-3):
        df[i] = np.sum(cent_coeff * f[i-3:i+4])

    back_coeff = -np.flip(forw_coeff)
    for i in range(n-3,n):
        df[i] = np.sum(back_coeff * f[i-6:i+1])

    return df


def fdiff2_7p(f):
    """7-points 2nd finite differences"""
    n = len(f)
    df = np.zeros_like(f)

    forw_coeff = np.array([203/45, -87/5, 117/4, -254/9, 33/2, -27/5, 137/180])
    for i in range(3):
        df[i] = np.sum(forw_coeff * f[i:i+7])

    cent_coeff = np.array([1/90, -3/20, 3/2, -49/18, 3/2, -3/20, 1/90])
    for i in range(3,n-3):
        df[i] = np.sum(cent_coeff * f[i-3:i+4])

    back_coeff = +np.flip(forw_coeff)
    for i in range(n-3,n):
        df[i] = np.sum(back_coeff * f[i-6:i+1])

    return df


def fsbt(l, f_g, r_g, G_k):
    """Fast spherical Bessel transform"""

    N = (len(G_k) - 1) * 2
    assert N == len(f_g)
    assert N == len(r_g)

    f_k = 0.0
    F_g = f_g * r_g

    for n in range(l + 1):
        f_k += (r_g[1] * (1j)**(l + 1 - n) *
                fac(l + n) / fac(l - n) / fac(n) / 2**n *
                np.fft.rfft(F_g, N)).real * G_k**(l - n)
        F_g[1:] /= r_g[1:]

    f_k[1:] /= G_k[1:]**(l + 1)

    if l == 0:
        f_k[0] = np.dot(r_g, f_g * r_g) * r_g[1]

    return f_k
