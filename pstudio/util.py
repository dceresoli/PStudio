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

import sys

# global output file and helper function
_out = None

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
