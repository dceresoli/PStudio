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
"""Generate pseudopotentials"""

from .util import find_rc_ic

def calculate_vpot(vae, rgd, rc, l, ene, pswfc, d2pswfc):
    """Generate the potential by inverting the Schroedinger equation"""

    # find the effective rc
    rc, ic = find_rc_ic(rgd, rc)

    # construct vpot
    vpot = vae.copy()
    vpot[:ic] = ene - l*(l+1)/(2.0*rgd.r[:ic]**2) + 0.5*d2pswfc[:ic]/pswfc[:ic]

    return vpot
