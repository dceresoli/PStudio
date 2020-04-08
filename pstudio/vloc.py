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
"""Generate a local potential"""
import numpy as np
from scipy.optimize import bisect, newton
from scipy.special import spherical_jn
from math import log, sin, cos, sqrt

from .util import find_rc_ic, calc_ae_norm, calc_ae_deriv
from .util import find_qi, dlog_bessel  #, deriv1, deriv2
from .util import p


def generate_vloc_RRKJ(vae, rgd, rc=None, verbose=False):
    """Genetate the local potential by pseudizing the AE potential with two Bessel functions"""

    # find the effective rc, calc AE norm and AE derivatives
    rc, ic = find_rc_ic(rgd, rc)
    ae_norm = calc_ae_norm(vae, rgd, ic)
    ae_deriv = calc_ae_deriv(vae, rgd, rc, ic, 3)
    if verbose:
        p('Local potential from RRKJ2 pseudization: rc={0:.4f}'.format(rc))
        p('AE norm within rc       : {0:+.6f}'.format(ae_norm))
        for i,d in enumerate(ae_deriv):
            p('{0}-th AE derivative at rc: {1:+.6f}'.format(i, d))

    # find q_i such that [jl(qi*rc)]'/jl(qi*rc) = phi'(rc)/phi(rc)
    ae_dlog = ae_deriv[1]/ae_deriv[0]
    l = 0
    fqi = lambda q: dlog_bessel(l, q, rc) - ae_dlog
    qi = find_qi(2, fqi)
    if verbose:
        p('qi               : ', qi)
        p('estimated cutoff : {0:g} Ha'.format(0.5*qi[-1]**2))

    # construct the linear system: two equations, two conditions
    lhs = np.zeros((2,2))
    lhs[0,:] = np.array([spherical_jn(l,qi[i]*rc) for i in range(2)])
    lhs[1,:] = np.array([deriv2(lambda x: spherical_jn(l,qi[i]*x), rc) for i in range(2)])

    # then the left hand side
    rhs = np.array([ae_deriv[0], ae_deriv[2]])

    # solve the linear system
    c = np.linalg.solve(lhs, rhs)

    # construct vloc
    vloc = vae.copy()
    vloc[:ic] = c[0]*spherical_jn(l,qi[0]*rgd.r[:ic]) \
               +c[1]*spherical_jn(l,qi[1]*rgd.r[:ic])
    p()

    return vloc


def generate_vloc_TM(vae, rgd, rc=None, verbose=False):
    """Genetate the local potential by pseudizing the AE potential with the TM method"""

    # find the effective rc, calc AE norm and AE derivatives
    rc, ic = find_rc_ic(rgd, rc)
    ae_norm = calc_ae_norm(vae, rgd, ic)
    ae_deriv = calc_ae_deriv(vae, rgd, rc, ic, 3)
    if verbose:
        p('Local potential from RRKJ2+TM pseudization: rc={0:.4f}'.format(rc))
        p('AE norm within rc       : {0:+.6f}'.format(ae_norm))
        for i,d in enumerate(ae_deriv):
            p('{0}-th AE derivative at rc: {1:+.6f}'.format(i, d))

    # find q_i such that [rc*jl(qi*rc)]"/[jl(qi*rc)]' = (rc*phi(rc))"/phi(rc)'
    ae2_dlog = rc * ae_deriv[2]/ae_deriv[1]
    l = 0
    f = lambda x: (-sin(x) + 2*sqrt(2)*sin(sqrt(2)*x/2))/x
    fp = lambda x: (x*(-cos(x) + 2*cos(sqrt(2)*x/2)) + sin(x) - 2*sqrt(2)*sin(sqrt(2)*x/2))/x**2
    fpp = lambda x: (x**2*(sin(x) - sqrt(2)*sin(sqrt(2)*x/2)) + 2*x*(cos(x) - 2*cos(sqrt(2)*x/2)) - 2*sin(x) + 4*sqrt(2)*sin(sqrt(2)*x/2))/x**2
    fqi = lambda q: (rc*q)*fpp(rc*q)/fp(rc*q) - ae2_dlog
    qi = find_qi(1, fqi)[0]
    if verbose:
        p('qi               : ', qi)
        p('estimated cutoff : {0:g} Ha'.format(0.5*qi**2))

    B = ae_deriv[1] / (qi * fp(qi*rc))
    A = ae_deriv[0] - B*f(qi*rc)

    # construct vloc
    vloc = vae.copy()
    for i in range(ic):
        vloc[i] = A + B * f(qi*rgd.r[i])
    p()

    return vloc
