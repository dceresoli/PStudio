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
"""RRKJ pseudization like in VASP and QE"""
import numpy as np
from scipy.optimize import bisect, newton
from scipy.special import spherical_jn
from math import log

from .util import p


def pseudize_RRKJ(fae, l, rc, rgd, nbess=3, rho0=0.1, verbose=False, plot_c2=False, c2=0.0):
    """Pseudize a radial function using the RRKJ method"""
    assert nbess == 3 or nbess == 4

    # find the effective rc
    ic = rgd.floor(rc)
    r = rgd.r
    rc = r[ic]
    if verbose:
        p('RRKJ{2} pseudization: l={1} rc={0:.4f}'.format(rc, l, nbess))

    # calculate norm of AE wfc within rc
    ae_norm = np.sum(fae[:ic]*fae[:ic] * rgd.dr[:ic])
    if verbose:
        p('AE norm within rc       : {0:+.6f}'.format(ae_norm))

    # calculate the derivatives of AE wfc
    poly = np.polyfit(r[ic-10:ic+10], fae[ic-10:ic+10], deg=6)
    ae_deriv = [np.polyval(np.polyder(poly,i),rc) for i in range(3)]
    ae_dlog = np.polyval(np.polyder(poly,1), rc) / np.polyval(poly, rc)
    if verbose:
        for i,d in enumerate(ae_deriv):
            p('{0}-th AE derivative at rc: {1:+.6f}'.format(i, d))

    # find q_i
    qi = find_qi(l, rc, ae_dlog, nbess)
    if verbose:
        p('qi: ', qi)
        p('estimated cutoff: {0:g} Ha'.format(0.5*qi[-1]**2))


    # experts only: plot the TM resisual as a function of c2 (there are two solutions!)
    if plot_c2:
        import matplotlib.pyplot as plt
        c2range = np.linspace(-10,10,1000)
        diff = np.zeros_like(c2range)
        for i in range(len(c2range)):
            diff[i] = RRKJ_calc_residual(c2range[i], qi, ae_norm, ae_deriv, rgd, rc, l)
        plt.gca().plot(c2range, diff)
        plt.gca().grid()

    # find RRKJ coefficients
    c2 = newton(lambda x: RRKJ_calc_residual(x, qi, ae_norm, ae_deriv, rgd, rc, l), x0=c2)

    # coefficient found
    c = RRKJ_solve_linear_problem(c2, qi, ae_deriv, rc, l)
    if verbose:
        p('RRKJ coefficients:', c)
        p('norm error       :', np.sum(RRKJ_function(r[:ic], l, c, qi)**2 * rgd.dr[:ic]) - ae_norm)

    # return pseudized function
    pswfc = fae.copy()
    pswfc[:ic] = RRKJ_function(r[:ic], l, c, qi)

    return pswfc


def dlog_bessel(l, q, r):
    return deriv1(lambda x: x*spherical_jn(l,q*x), r) / (r * spherical_jn(l,q*r))

def deriv1(f, x, dx=0.001):
    return (f(x+dx)-f(x-dx))/(2*dx)

def deriv2(f, x, dx=0.001):
    return (f(x+dx)-2*f(x)+f(x-dx))/(dx*dx)


def find_qi(l, rc, ae_dlog, nbess):
    # find all possible q_i's
    qrange = np.linspace(0.01, 20, 100)
    qi = []
    for i in range(len(qrange)-1):
        try:
            q0 = bisect(lambda q: dlog_bessel(l,q,rc)-ae_dlog, a=qrange[i], b=qrange[i+1])
        except ValueError:
            pass
        else:
            if abs(dlog_bessel(l, q0, rc)) < 100:  # eliminate asymptotes
                qi.append(q0)
        # exit when found all q_i's
        if len(qi) == nbess:
            break

    return np.array(qi)


def RRKJ_function(r, l, c, qi):
    """Evaluate the RRKJ pseudowfc"""
    res = 0.0
    for i in range(len(qi)):
        res += c[i]*spherical_jn(l, r*qi[i])
    return r * res


def RRKJ_linear_problem(c2, qi, ae_deriv, rc, l):
    """Construct the RRKJ linear problem as a function of c2"""
    if len(qi) == 4:
        raise NotImplementedError

    # first the left hand side
    lhs = np.zeros((3,3))
    lhs[0,:] = np.array([rc*spherical_jn(l,qi[i]*rc) for i in range(3)])
    lhs[1,:] = np.array([deriv1(lambda x: x*spherical_jn(l,qi[i]*x), rc) for i in range(3)])
    lhs[2,:] = np.array([deriv2(lambda x: x*spherical_jn(l,qi[i]*x), rc) for i in range(3)])

    # then the left hand side
    rhs = ae_deriv[0:3]

    # eliminate the second equation because it's linear dependent with the 1st
    lhs = np.delete(lhs, (1), axis=0)
    rhs = np.delete(rhs, (1))

    # eliminate the column of c2 and move it to the rhs
    rhs -= c2*lhs[:,2]
    lhs = np.delete(lhs, (2), axis=1)
    return lhs, rhs


def RRKJ_solve_linear_problem(c2, qi, ae_deriv, rc, l):
    # solve linear part of the system
    lhs, rhs = RRKJ_linear_problem(c2, qi, ae_deriv, rc, l)
    c = np.linalg.solve(lhs, rhs)

    # put back c2 into the list
    c = list(c)
    c.append(c2)

    return c


def RRKJ_calc_residual(c2, qi, ae_norm, ae_deriv, rgd, rc, l):
    # solve linear part of the system
    c = RRKJ_solve_linear_problem(c2, qi, ae_deriv, rc, l)

    # fix norm-conserving relation
    g = rgd.floor(rc)
    r = rgd.r[:g]
    ps_norm = np.sum(RRKJ_function(r, l, c, qi)**2 * rgd.dr[:g])
    diff = ps_norm - ae_norm

    return diff
