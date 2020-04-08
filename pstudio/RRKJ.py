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
#from scipy.special import spherical_jn
from math import log

from .util import find_rc_ic, calc_ae_norm, calc_ae_deriv
from .util import find_qi, dlog_rbessel, rqbess, rqbessp, rqbesspp
from .util import p

def pseudize_RRKJ(fae, l, rc, rgd, nbess=3, rho0=0.1, verbose=False, plot_c2=False, c2=0.0):
    """Pseudize a radial function using the RRKJ method"""
    assert nbess == 3 or nbess == 4

    # find the effective rc, calc AE norm and AE derivatives
    rc, ic = find_rc_ic(rgd, rc)
    ae_norm = calc_ae_norm(fae, rgd, ic)
    ae_deriv = calc_ae_deriv(fae, rgd, rc, ic, 3)
    if verbose:
        p('RRKJ{2} pseudization: l={1} rc={0:.4f}'.format(rc, l, nbess))
        p('AE norm within rc       : {0:+.6f}'.format(ae_norm))
        for i,d in enumerate(ae_deriv):
            p('{0}-th AE derivative at rc: {1:+.6f}'.format(i, d))

    # find q_is uch that [rc*jl(qi*rc)]'/(rc*jl(qi*rc)) = phi'(rc)/phi(rc)
    ae_dlog = ae_deriv[1]/ae_deriv[0]
    fqi = lambda q: dlog_rbessel(l, q, rc) - ae_dlog
    qi = find_qi(nbess, fqi)
    if verbose:
        p('qi               : ', qi)
        p('estimated cutoff : {0:g} Ha'.format(0.5*qi[-1]**2))

    # experts only: plot the TM resisual as a function of c2 (there are two solutions!)
    # TODO: write a functions just for that
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
        p('norm error       :', np.sum(RRKJ_function(rgd.r[:ic], l, c, qi)**2 * rgd.dr[:ic]) - ae_norm)
        p()

    # return pseudized function and it's 2nd derivative to calcualte the pseudopotential
    pswfc = fae.copy()
    pswfc[:ic] = RRKJ_function(rgd.r[:ic], l, c, qi)
    d2pswfc = rgd.deriv2(fae)
    d2pswfc[:ic] = RRKJ_function_pp(rgd.r[:ic], l, c, qi)

    return pswfc, d2pswfc


def RRKJ_function(r, l, c, qi):
    """Evaluate the RRKJ pseudowfc"""
    res = 0.0
    for i in range(len(qi)):
        res += c[i] * r * spherical_jn(l, r*qi[i])
    return res

def RRKJ_function_pp(r, l, c, qi):
    """Evaluate the 2nd derivative of the RRKJ pseudowfc"""
    res = 0.0
    for i in range(len(qi)):
        f = lambda r: r*spherical_jn(l, r*qi[i])
        res += c[i]*deriv2(f, r)
    return res

def RRKJ_linear_problem(c2, qi, ae_deriv, rc, l):
    """Construct the RRKJ linear problem as a function of c2"""
    if len(qi) == 4:
        raise NotImplementedError

    # first the left hand side
    lhs = np.zeros((3,3))
    #lhs[0,:] = np.array([rc*spherical_jn(l,qi[i]*rc) for i in range(3)])
    #lhs[1,:] = np.array([deriv1(lambda x: x*spherical_jn(l,qi[i]*x), rc) for i in range(3)])
    #lhs[2,:] = np.array([deriv2(lambda x: x*spherical_jn(l,qi[i]*x), rc) for i in range(3)])
    lhs[0,:] = np.array([rqbess(l, qi[i], rc) for i in range(3)])
    lhs[0,:] = np.array([rqbessp(l, qi[i], rc) for i in range(3)])
    lhs[0,:] = np.array([rqbesspp(l, qi[i], rc) for i in range(3)])

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
