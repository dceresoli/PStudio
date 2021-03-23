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
"""Trouiller-Martin pseudization"""
import numpy as np
from scipy.optimize import newton
from math import log

from .util import find_rc_ic, calc_ae_norm, calc_ae_deriv
from .util import p


def pseudize_TM(fae, l, rc, rgd, verbose=False, plot_c2=False, c2=0.0):
    """Pseudize a radial function using the TM method"""

    # find the effective rc, calc AE norm and AE derivatives
    rc, ic = find_rc_ic(rgd, rc)
    ae_norm = calc_ae_norm(fae, rgd, ic)
    ae_deriv = calc_ae_deriv(fae, rgd, rc, ic, 5)
    if verbose:
        p('TM pseudization: l={1} rc={0:.4f}'.format(rc, l))
        p('AE norm within rc       : {0:+.6f}'.format(ae_norm))
        for i,d in enumerate(ae_deriv):
            p('{0}-th AE derivative at rc: {1:+.6f}'.format(i, d))

    # experts only: plot the TM resisual as a function of c2 (there are two solutions!)
    # TODO: write a functions just for that
    if plot_c2:
        import matplotlib.pyplot as plt
        c2range = np.linspace(-10,10,1000)
        diff = np.zeros_like(c2range)
        for i in range(len(c2range)):
            diff[i] = TM_calc_residual(c2range[i], ae_norm, ae_deriv, rgd, rc, l)
        plt.gca().plot(c2range, diff)
        plt.gca().grid()

    # find TM coefficients
    c2 = newton(lambda x: TM_calc_residual(x, ae_norm, ae_deriv, rgd, rc, l), x0=c2)

    # coefficients found
    c = TM_solve_linear_problem(c2, ae_deriv, rc, l)
    if verbose:
        p('TM coefficients:', c)
        p('norm error     :', np.sum(TM_function(rgd.r[:ic], l, c)**2 * rgd.dr[:ic])- ae_norm)
        p('V"(0) condition:', (2*l+5)*c[2] + c[1]*c[1])
        p()

    # return pseudized function and it's 2nd derivative to calcualte the pseudopotential
    pswfc = fae.copy()
    pswfc[:ic] = TM_function(rgd.r[:ic], l, c)
    d2pswfc = rgd.spline_deriv2(fae)
    d2pswfc[:ic] = TM_function_pp(rgd.r[:ic], l, c)

    return pswfc, d2pswfc


def TM_function(r, l, c):
    """Evaluate the TM pseudowfc"""
    c0, c2, c4, c6, c8, c10, c12 = c
    poly = np.array([c12,0,c10,0,c8,0,c6,0,c4,0,c2,0,c0])
    return r**(l+1) * np.exp(np.polyval(poly,r))

def TM_function_pp(r, l, c):
    """Evaluate the 2nd derivateice of the TM pseudowfc"""
    c0, c2, c4, c6, c8, c10, c12 = c
    poly = np.array([c12,0,c10,0,c8,0,c6,0,c4,0,c2,0,c0])
    polyp = np.polyder(poly)
    polypp = np.polyder(polyp)
    phi = r**(l+1) * np.exp(np.polyval(poly,r))
    return phi * ( l*(l+1)/r**2 + 2*(l+1)/r*np.polyval(polyp,r) \
                   + np.polyval(polyp,r)**2 + np.polyval(polypp,r) )

def TM_linear_problem(c2, ae_deriv, rc, l):
    """Construct the TM linear problem as a function of c2"""
    # first the left hand side
    lhs = np.zeros((6,7))
    p = np.array([0,2,4,6,8,10,12])   # powers
    c = np.array([1,1,1,1,1,1,1])     # coefficients
    for i in range(5):
        lhs[i,:] = c * rc**p
        c = c*p
        p = np.array([max(0,p[j]-1) for j in range(7)])
    lhs[5,2] = 2*l + 5                # coefficient of c4

    # then the left hand side
    rhs = np.zeros(6)
    rhs[0] = log(ae_deriv[0]/rc**(l+1))
    rhs[1] = -(l+1)/rc + ae_deriv[1]/ae_deriv[0]
    rhs[2] = (l+1)/rc**2 + ae_deriv[2]/ae_deriv[0] \
             - ae_deriv[1]**2/ae_deriv[0]**2
    rhs[3] = -2*(l+1)/rc**3 + ae_deriv[3]/ae_deriv[0] \
             - 3*ae_deriv[1]*ae_deriv[2]/ae_deriv[0]**2 \
             + 2*ae_deriv[1]**3/ae_deriv[0]**3
    rhs[4] = +6*(l+1)/rc**4 + ae_deriv[4]/ae_deriv[0] \
             - 4*ae_deriv[1]*ae_deriv[3]/ae_deriv[0]**2 \
             - 3*ae_deriv[2]**2/ae_deriv[0]**2 \
             +12*ae_deriv[1]**2*ae_deriv[2]/ae_deriv[0]**3 \
             - 6*ae_deriv[1]**4/ae_deriv[0]**4

    # eliminate the column of c2 and move it to the rhs
    rhs -= c2*lhs[:,1]
    lhs = np.delete(lhs, (1), axis=1)
    rhs[5] = -c2*c2
    return lhs, rhs


def TM_solve_linear_problem(c2, ae_deriv, rc, l):
    # solve linear part of the system
    lhs, rhs = TM_linear_problem(c2, ae_deriv, rc, l)
    c = np.linalg.solve(lhs, rhs)

    # put back c2 into the list
    c = list(c)
    c.insert(1, c2)
    return c


def TM_calc_residual(c2, ae_norm, ae_deriv, rgd, rc, l):
    # solve linear part of the system
    c = TM_solve_linear_problem(c2, ae_deriv, rc, l)

    # fix norm-conserving relation
    ic = rgd.floor(rc)
    r = rgd.r[:ic]
    ps_norm = np.sum(TM_function(r, l, c)**2 * rgd.dr[:ic])
    diff = ps_norm - ae_norm

    return diff
