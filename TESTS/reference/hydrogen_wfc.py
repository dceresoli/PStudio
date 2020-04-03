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
Generate hydrogenoic wavefunctions
"""

import math
import numpy as np

# helper functions
def binomial(n, m):
    return math.factorial(n)//(math.factorial(n-m)*math.factorial(m))

def gen_laguerre(n, alpha, x):
    res = np.zeros_like(x)
    for i in range(n+1):
       res += (-1)**i * binomial(n+alpha,n-i) * (x**i) / math.factorial(i)
    return res

# unnormalized hydrogenoic wave function
def hydrogen_wfc(r, zeta, n, l):
    rho = 2.0*r*zeta/n
    psi = np.exp(-rho/2) * (rho**(l+1)) * gen_laguerre(n-l-1, 2*l+1, rho)
    return psi
