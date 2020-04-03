# AtomPy - atomic calcalations in python
# Copyright (C) 2010  Davide Ceresoli <dceresoli@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation version 3 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
Exchange and correlation functionals.
"""
from math import log, exp, sqrt

# The x_... and c_... routines return epsilon_XC(rho), not E_XC(rho(r))
# XC energy density and XC potential are in Hartree

pi34 = 0.6203504908994  # (3.0/(4.0*pi))**(1.0/3.0)
third = 1.0/3.0
        
# funcionals
class LDA:
    name = "LDA"
    full_name = "LDA (PZ parametrization)"
    citation = "J. P. Perdew and A. Zunger, PRB 23, 5048 (1981)"

    @staticmethod
    def xc(rho):
        rs = pi34 / rho**third
        return x_slater(rs), c_pz(rs)



# common parametrizations
def x_slater(rs):
    """Slater exchange"""
    f = -0.687247939924714  # -(9.0/8.0)*(3.0/(2.0*pi))**(2.0/3.0)
    alpha = 2.0/3.0

    ex = f * alpha / rs
    vx = 4.0/3.0 * f * alpha / rs
    return ex, vx  # rydberg or hartree?


def c_pz(rs):
    """PZ correlation"""
    a, b, c, d = 0.0311, -0.048, 0.002, -0.0116
    gc, b1, b2 = -0.1423, 1.0529, 0.3334
    if rs < 1.0:
        # high density formula
        lnrs = log(rs)
        ec = a*lnrs + b + c*rs*lnrs + d*rs
        vc = a*lnrs + (b-a/3.0) + (2.0/3.0) * c*rs*lnrs + (2.0*d-c)*rs/3.0
    else:
        # interpolation formula
        rs12 = sqrt(rs)
        ox = 1.0 + b1*rs12 + b2*rs
        dox = 1.0 + (7.0/6.0)*b1*rs12 + 4.0/3.0*b2*rs
        ec = gc / ox
        vc = ec * dox / ox
    return ec, vc


if __name__ == '__main__':
    print LDA.name, LDA.full_name
    rho = 1e-2
    while rho < 10:
        (ex, vx), (ec, vc) = LDA.xc(rho)
        rs = pi34 / rho**third
        print rs, ex, ec
        rho *= 1.1





