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
"""
Periodic table of elements: mass, atomic number, name and electronic
configuration.
"""
import sys

# TODO: create core-corehole


def parse_configuration(conf):
    """Return a list of (n,l,occ) tuples from a string"""
    spdf_to_l = {'s': 0, 'p': 1, 'd':2, 'f': 3}

    # expand rare gases configuration
    while conf[0] == '[':
        atom = conf[1:3]
        conf = atom_table[atom][3] + conf[4:]

    # split string
    config = []
    for t in conf.split():
        n = int(t[0])
        l = spdf_to_l[t[1]]
        occ = float(t[2:])
        config.append((n, l, occ))
    return config


def tuple_to_configuration(conf):
    """Return a string from a list of (n,l,occ) tuples"""
    l_to_spdf = 'spdf'
    config = ' '.join(['{0}{1}{2:g}'.format(orb[0], l_to_spdf[orb[1]], orb[2]) \
             for orb in conf])
    return config


class Element:
    def __init__(self, symbol):
        self.symbol = symbol
        try:
            self.data = atom_table[symbol]
        except KeyError:
            raise RuntimeError('element {0} is unknown'.format(symbol))

    def get_name(self):
        return self.data[0]

    def get_atomic_number(self):
        return self.data[1]

    def get_mass(self):
        return self.data[2]

    def get_configuration(self):
        return self.data[3]


# symbol: [ name, number, mass, configuration ]
atom_table = {
    'H' : ['Hydrogen',    1, 1.01,  '1s1'],
    'He': ['Helium',      2, 4,     '1s2'],
    'Li': ['Lithium',     3, 6.94,  '[He] 2s1'],
    'Be': ['Beryllium',   4, 9.01,  '[He] 2s2'],
    'B' : ['Boron',       5, 10.81, '[He] 2s2 2p1'],
    'C' : ['Carbon',      6, 12.01, '[He] 2s2 2p2'],
    'N' : ['Nitrogen',    7, 14.01, '[He] 2s2 2p3'],
    'O' : ['Oxygen',      8, 16,    '[He] 2s2 2p4'],
    'F' : ['Fluorine',    9, 19,    '[He] 2s2 2p5'],
    'Ne': ['Neon',       10, 20.18, '[He] 2s2 2p6'],
    'Na': ['Sodium',     11, 22.99, '[Ne] 3s1'],
    'Mg': ['Magnesium',  12, 24.31, '[Ne] 3s2'],
    'Al': ['Aluminum',   13, 26.98, '[Ne] 3s2 3p1'],
    'Si': ['Silicon',    14, 28.09, '[Ne] 3s2 3p2'],
    'P' : ['Phosphorus', 15, 30.97, '[Ne] 3s2 3p3'],
    'S' : ['Sulfur',     16, 32.07, '[Ne] 3s2 3p4'],
    'Cl': ['Chlorine',   17, 35.45, '[Ne] 3s2 3p5'],
    'Ar': ['Argon',      18, 39.95, '[Ne] 3s2 3p6'],
    'K' : ['Potassium',  19, 39.10, '[Ar] 4s1'],
    'Ca': ['Calcium',    20, 40.08, '[Ar] 4s2'],
    'Sc': ['Scandium',   21, 44.96, '[Ar] 3d1 4s2'],
    'Ti': ['Titanium',   22, 47.87, '[Ar] 3d2 4s2'],
    'V' : ['Vanadium',   23, 50.94, '[Ar] 3d3 4s2'],
    'Cr': ['Chromium',   24, 52.00, '[Ar] 3d5 4s1'],
    'Mn': ['Manganese',  25, 54.94, '[Ar] 3d5 4s2'],
    'Fe': ['Iron',       26, 55.85, '[Ar] 3d6 4s2'],
    'Co': ['Cobalt',     27, 58.93, '[Ar] 3d7 4s2'],
    'Ni': ['Nickel',     28, 58.69, '[Ar] 3d8 4s2'],
    'Cu': ['Copper',     29, 63.55, '[Ar] 3d10 4s1'],
    'Zn': ['Zinc',       30, 65.39, '[Ar] 3d10 4s2'],
    'Ga': ['Gallium',    31, 69.72, '[Ar] 3d10 4s2 4p1'],
    'Ge': ['Germanium',  32, 72.64, '[Ar] 3d10 4s2 4p2'],
    'As': ['Arsenic',    33, 74.92, '[Ar] 3d10 4s2 4p3'],
    'Se': ['Selenium',   34, 78.96, '[Ar] 3d10 4s2 4p4'],
    'Br': ['Bromine',    35, 79.90, '[Ar] 3d10 4s2 4p5'],
    'Kr': ['Krypton',    36, 83.80, '[Ar] 3d10 4s2 4p6'],
    'Rb': ['Rubidium',   37, 85.47, '[Kr] 5s1'],
    'Sr': ['Strontium',  38, 87.62, '[Kr] 5s2'],
    'Y' : ['Yttrium',    39, 88.91, '[Kr] 4d1 5s2'],
    'Zr': ['Zirconium',  40, 91.22, '[Kr] 4d2 5s2'],
    'Nb': ['Niobium',    41, 92.91, '[Kr] 4d4 5s1'],
    'Mo': ['Molybdenum', 42, 95.94, '[Kr] 4d5 5s1'],
    'Tc': ['Technetium', 43, 98.00, '[Kr] 4d5 5s2'],
    'Ru': ['Ruthenium',  44, 101.07, '[Kr] 4d7 5s1'],
    'Rh': ['Rhodium',    45, 102.91, '[Kr] 4d8 5s1'],
    'Pd': ['Palladium',  46, 106.42, '[Kr] 4d10'],
    'Ag': ['Silver',     47, 107.87, '[Kr] 4d10 5s1'],
    'Cd': ['Cadmium',    48, 112.41, '[Kr] 4d10 5s2'],
    'In': ['Indium',     49, 114.82, '[Kr] 4d10 5s2 5p1'],
    'Sn': ['Tin',        50, 118.71, '[Kr] 4d10 5s2 5p2'],
    'Sb': ['Antimony',   51, 121.76, '[Kr] 4d10 5s2 5p3'],
    'Te': ['Tellurium',  52, 127.60, '[Kr] 4d10 5s2 5p4'],
    'I' : ['Iodine',     53, 126.90, '[Kr] 4d10 5s2 5p5'],
    'Xe': ['Xenon',      54, 131.29, '[Kr] 4d10 5s2 5p6'],
    'Cs': ['Cesium',     55, 132.91, '[Xe] 6s1'],
    'Ba': ['Barium',     56, 137.33, '[Xe] 6s2'],
    'La': ['Lanthanum',  57, 138.91, '[Xe] 5d1 6s2'],
    'Ce': ['Cerium',     58, 140.12, '[Xe] 4f1 5d1 6s2'],
    'Pr': ['Praseodymium', 59, 140.91, '[Xe] 4f3 6s2'],
    'Nd': ['Neodymium',  60, 144.24, '[Xe] 4f4 6s2'],
    'Pm': ['Promethium', 61, 145.00, '[Xe] 4f5 6s2'],
    'Sm': ['Samarium',   62, 150.36, '[Xe] 4f6 6s2'],
    'Eu': ['Europium',   63, 151.96, '[Xe] 4f7 6s2'],
    'Gd': ['Gadolinium', 64, 157.25, '[Xe] 4f7 5d1 6s2'],
    'Tb': ['Terbium',    65, 158.93, '[Xe] 4f9 6s2'],
    'Dy': ['Dysprosium', 66, 162.5,  '[Xe] 4f10 6s2'],
    'Ho': ['Holmium',    67, 164.93, '[Xe] 4f11 6s2'],
    'Er': ['Erbium',     68, 167.26, '[Xe] 4f12 6s2'],
    'Tm': ['Thulium',    69, 168.93, '[Xe] 4f13 6s2'],
    'Yb': ['Ytterbium',  70, 173.04, '[Xe] 4f14 6s2'],
    'Lu': ['Lutetium',   71, 174.97, '[Xe] 4f14 5d1 6s2'],
    'Hf': ['Hafnium',    72, 178.49, '[Xe] 4f14 5d2 6s2'],
    'Ta': ['Tantalum',   73, 180.95, '[Xe] 4f14 5d3 6s2'],
    'W' : ['Tungsten',   74, 183.84, '[Xe] 4f14 5d4 6s2'],
    'Re': ['Rhenium',    75, 186.21, '[Xe] 4f14 5d5 6s2'],
    'Os': ['Osmium',     76, 190.23, '[Xe] 4f14 5d6 6s2'],
    'Ir': ['Iridium',    77, 192.22, '[Xe] 4f14 5d7 6s2'],
    'Pt': ['Platinum',   78, 195.08, '[Xe] 4f14 5d9 6s1'],
    'Au': ['Gold',       79, 196.97, '[Xe] 4f14 5d10 6s1'],
    'Hg': ['Mercury',    80, 200.59, '[Xe] 4f14 5d10 6s2'],
    'Tl': ['Thallium',   81, 204.38, '[Xe] 4f14 5d10 6s2 6p1'],
    'Pb': ['Lead',       82, 207.20, '[Xe] 4f14 5d10 6s2 6p2'],
    'Bi': ['Bismuth',    83, 208.98, '[Xe] 4f14 5d10 6s2 6p3'],
    'Po': ['Polonium',   84, 209,    '[Xe] 4f14 5d10 6s2 6p4'],
    'At': ['Astatine',   85, 210,    '[Xe] 4f14 5d10 6s2 6p5'],
    'Rn': ['Radon',      86, 222,    '[Xe] 4f14 5d10 6s2 6p6'],
    'Fr': ['Francium',   87, 223,    '[Rn] 7s1'],
    'Ra': ['Radium',     88, 226,    '[Rn] 7s2'],
    'Ac': ['Actinium',   89, 227,    '[Rn] 6d1 7s2'],
    'Th': ['Thorium',    90, 232.04, '[Rn] 6d2 7s2'],
    'Pa': ['Protactinium', 91, 231.04, '[Rn] 5f2 6d1 7s2 '],
    'U' : ['Uranium',    92, 238.03, '[Rn] 5f3 6d1 7s2'],
    'Np': ['Neptunium',  93, 237,    '[Rn] 5f4 6d1 7s2'],
    'Pu': ['Plutonium',  94, 244,    '[Rn] 5f6 7s2'],
    'Am': ['Americium',  95, 243,    '[Rn] 5f7 7s2']
}


if __name__ == '__main__':
    for symbol in atom_table:
        el = Element(symbol)
        config = parse_configuration(el.get_configuration())
        nelec = 0
        for orbital in config:
            nelec += orbital[2]
        print(symbol, el.get_atomic_number(), nelec)
