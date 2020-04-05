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

def dlog_rbessel(l, q, r):
    """log derivative: (r*q * j_l(r*q))' / (r*q * j_l(r*q))"""
    return deriv1(lambda x: x*spherical_jn(l,q*x), r) / (r * spherical_jn(l,q*r))

def dlog_bessel(l, q, r):
    """log derivative: (j_l(r*q))' / (j_l(r*q))"""
    return deriv1(lambda x: spherical_jn(l,q*x), r) / spherical_jn(l,q*r)

def find_qi(l, rc, ae_dlog, nbess, rflag=True):
    """find all possible q_i's to match the AE log der"""
    qrange = np.linspace(0.05, 20, 100)
    qi = []

    if rflag:
        f = dlog_rbessel
    else:
        f = dlog_bessel

    for i in range(len(qrange)-1):
        try:
            q0 = bisect(lambda q: f(l,q,rc)-ae_dlog, a=qrange[i], b=qrange[i+1])
        except ValueError:
            pass
        else:
            if abs(f(l, q0, rc)) < 100:  # eliminate asymptotes
                qi.append(q0)

        # exit when found all q_i's
        if len(qi) == nbess:
            break

    return np.array(qi)


# numerical derivatives of a real function
def deriv1(f, x, dx=0.001):
    return (f(x+dx)-f(x-dx))/(2*dx)

def deriv2(f, x, dx=0.001):
    return (f(x+dx)-2*f(x)+f(x-dx))/(dx*dx)
