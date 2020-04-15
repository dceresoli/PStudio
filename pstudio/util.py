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
"""Utility functions"""

import numpy as np
from scipy.optimize import bisect
from scipy.special import spherical_jn
import sys


# physical constants
hartree = 27.211386        # in #!/usr/bin/env python
alpha = 1/137.035999084    # fine structure constant


# global output file and helper function
_out = sys.stdout

def set_output(out):
    """Set the default output file or stream"""
    global _out
    if out == '-':
        _out = sys.stdout
    elif isinstance(out, str):
        _out = open(out, 'w')
    else:
        _out = out

def p(*args, **kwargs):
    """Helper routine to output data to a file or to stdout"""
    global _out
    if _out is not None:
        print(*args, **kwargs, file=_out)


# inherit from this class to freeze class attributes
class frozen:
    __isfrozen = False
    def __setattr__(self, key, value):
        if self.__isfrozen and not hasattr(self, key):
            raise TypeError( "%r is a frozen class" % self )
        object.__setattr__(self, key, value)

    def _freeze(self):
        self.__isfrozen = True


# spherical Bessel functions and their derivatives
def qbess(l, q, r):
    x = q*r
    return spherical_jn(l,x)

def qbessp(l, q, r):
    x = q*r
    return q*spherical_jn(l,x,derivative=True)

def qbesspp(l, q, r):
    x = q*r
    a = (l*(l+1) - x*x)*spherical_jn(l,x) - 2*x*spherical_jn(l,x,derivative=True)
    return a * q*q / (x*x)

# r times spherical Bessel and their derivatives
def rqbess(l, q, r):
    return r*qbess(l, q, r)

def rqbessp(l, q, r):
    return qbess(l, q, r) + r*qbessp(l, q, r)

def rqbesspp(l, q, r):
    return 2*qbessp(l, q, r) + r*qbesspp(l, q, r)


# log deriv of soherical Bessel functions
def dlog_rbessel(l, q, r):
    """log derivative: (r*q * j_l(r*q))' / (r*q * j_l(r*q))"""
    return rqbessp(l, q, r) / rqbess(l, q, r)
    #return deriv1(lambda x: x*spherical_jn(l,q*x), r) / (r * spherical_jn(l,q*r))

def dlog_bessel(l, q, r):
    """log derivative: (j_l(r*q))' / (j_l(r*q))"""
    return qbessp(l, q, r) / qbess(l, q, r)
    #return deriv1(lambda x: spherical_jn(l,q*x), r) / spherical_jn(l,q*r)



# these routines are used both in TM and  RRKJ methods
def find_rc_ic(rgd, rc):
    """Find the effective rc and return it with ic, the index of the matching radius"""
    ic = rgd.floor(rc)
    rc = rgd.r[ic]
    return rc, ic

def calc_ae_norm(fae, rgd, ic):
    """Return the norm the AE function within ic"""
    return np.sum(fae[:ic]*fae[:ic] * rgd.dr[:ic])

def calc_ae_deriv(fae, rgd, rc, ic, nderiv):
    """Calculate the derivatives of the AE function at rc"""
    r = rgd.r
    poly = np.polyfit(r[ic-10:ic+10], fae[ic-10:ic+10], deg=6)
    ae_deriv = [np.polyval(np.polyder(poly,i),rc) for i in range(nderiv)]
    return np.array(ae_deriv)

def find_qi(nqi, fqi, qmax=20.0):
    """find all possible q_i's such that fqi is zero"""
    qrange = np.linspace(0.05, qmax, 100)

    qi = []
    for i in range(len(qrange)-1):
        try:
            q0 = bisect(fqi, a=qrange[i], b=qrange[i+1])
        except ValueError:
            pass
        else:
            if abs(fqi(q0)) < 100:  # eliminate asymptotes
                qi.append(q0)

        # exit when found all q_i's
        if len(qi) == nqi:
            break

    return np.array(qi)


# Thomas Fermi potential (from ONCVSPS src/tfaport.f90)
def thomas_fermi(z, r):
    """Thomas-Fermi potential"""
    b = (0.69395656/z)**(1.0/3.0)
    x = r / b
    xs = np.sqrt(x)

    t = z/(1.0+xs*(0.02747 - x*(0.1486 - 0.007298*x)) + x*(1.243 + x*(0.2302 + 0.006944*x)))

    t[t<1] = 1
    return -t/r


## numerical derivatives of a real function
#def deriv1(f, x, dx=0.001):
#    return (f(x+dx)-f(x-dx))/(2*dx)
#
#def deriv2(f, x, dx=0.001):
#    return (f(x+dx)-2*f(x)+f(x-dx))/(dx*dx)
