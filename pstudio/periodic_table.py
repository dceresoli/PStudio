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
"""Periodic table of elements: mass, atomic number, name and electronic configuration"""

class Element:
    """Hold atomic data"""
    def __init__(self, symbol):
        if isinstance(symbol, int):
            if symbol < 1 or symbol > _max_z:
                raise RuntimeError('element with z={0} not in database'.format(symbol))
            self._z = symbol
        else:
            if 1 <= len(symbol) <=2 and symbol in _atom_symbol:
                self._z = _atom_symbol.index(symbol) + 1
            elif symbol in _atom_name:
                self._z = _atom_name.index(symbol) + 1
            else:
                raise RuntimeError('element {0} not in database'.format(symbol))

        self._symbol = _atom_symbol[self._z - 1]
        self._name = _atom_name[self._z - 1]
        self._configuration = _atom_configuration[self._z - 1]
        self._mass = _atom_mass[self._z - 1]
        self._rcov = _atom_rcov[self._z - 1]

    @property
    def z(self): return self._z

    @property
    def symbol(self): return self._symbol

    @property
    def name(self): return self._name

    @property
    def configuration(self): return self._configuration

    @property
    def mass(self): return self._mass

    @property
    def rcov(self): return self._rcov


# routines to parse atomic configuration from a string and viceversa
# TODO: create core-corehole
def parse_configuration(conf):
    """Return a list of (n,l,occ) tuples from a string"""
    spdf_to_l = {'s': 0, 'p': 1, 'd':2, 'f': 3}

    # expand rare gases configuration
    while conf[0] == '[':
        atom = conf[1:3]
        conf = Element(atom).configuration + conf[4:]

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


_max_z = 95  # Americium
_atom_symbol = [
  'H', 'He',
  'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
  'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
  'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
  'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe',
  'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn',
  'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am',
]

_atom_name = [
  'Hydrogen', 'Helium', 'Lithium', 'Beryllium', 'Boron', 'Carbon',
  'Nitrogen', 'Oxygen', 'Fluorine', 'Neon', 'Sodium', 'Magnesium',
  'Aluminum', 'Silicon', 'Phosphorus', 'Sulfur', 'Chlorine', 'Argon',
  'Potassium', 'Calcium', 'Scandium', 'Titanium', 'Vanadium', 'Chromium',
  'Manganese', 'Iron', 'Cobalt', 'Nickel', 'Copper', 'Zinc', 'Gallium',
  'Germanium', 'Arsenic', 'Selenium', 'Bromine', 'Krypton', 'Rubidium',
  'Strontium', 'Yttrium', 'Zirconium', 'Niobium', 'Molybdenum',
  'Technetium', 'Ruthenium', 'Rhodium', 'Palladium', 'Silver', 'Cadmium',
  'Indium', 'Tin', 'Antimony', 'Tellurium', 'Iodine', 'Xenon', 'Cesium',
  'Barium', 'Lanthanum', 'Cerium', 'Praseodymium', 'Neodymium',
  'Promethium', 'Samarium', 'Europium', 'Gadolinium', 'Terbium',
  'Dysprosium', 'Holmium', 'Erbium', 'Thulium', 'Ytterbium', 'Lutetium',
  'Hafnium', 'Tantalum', 'Tungsten', 'Rhenium', 'Osmium', 'Iridium',
  'Platinum', 'Gold', 'Mercury', 'Thallium', 'Lead', 'Bismuth', 'Polonium',
  'Astatine', 'Radon', 'Francium', 'Radium', 'Actinium', 'Thorium',
  'Protactinium', 'Uranium', 'Neptunium', 'Plutonium', 'Americium',
]

_atom_mass = [
  1.01, 4, 6.94, 9.01, 10.81, 12.01, 14.01, 16, 19, 20.18, 22.99, 24.31,
  26.98, 28.09, 30.97, 32.07, 35.45, 39.95, 39.1, 40.08, 44.96, 47.87,
  50.94, 52.0, 54.94, 55.85, 58.93, 58.69, 63.55, 65.39, 69.72, 72.64,
  74.92, 78.96, 79.9, 83.8, 85.47, 87.62, 88.91, 91.22, 92.91, 95.94,
  98.0, 101.07, 102.91, 106.42, 107.87, 112.41, 114.82, 118.71, 121.76,
  127.6, 126.9, 131.29, 132.91, 137.33, 138.91, 140.12, 140.91, 144.24,
  145.0, 150.36, 151.96, 157.25, 158.93, 162.5, 164.93, 167.26, 168.93,
  173.04, 174.97, 178.49, 180.95, 183.84, 186.21, 190.23, 192.22, 195.08,
  196.97, 200.59, 204.38, 207.2, 208.98, 209, 210, 222, 223, 226, 227,
  232.04, 231.04, 238.03, 237, 244, 243,
]

_atom_configuration = [
  '1s1',
  '1s2',
  '[He] 2s1',
  '[He] 2s2',
  '[He] 2s2 2p1',
  '[He] 2s2 2p2',
  '[He] 2s2 2p3',
  '[He] 2s2 2p4',
  '[He] 2s2 2p5',
  '[He] 2s2 2p6',
  '[Ne] 3s1',
  '[Ne] 3s2',
  '[Ne] 3s2 3p1',
  '[Ne] 3s2 3p2',
  '[Ne] 3s2 3p3',
  '[Ne] 3s2 3p4',
  '[Ne] 3s2 3p5',
  '[Ne] 3s2 3p6',
  '[Ar] 4s1',
  '[Ar] 4s2',
  '[Ar] 3d1 4s2',
  '[Ar] 3d2 4s2',
  '[Ar] 3d3 4s2',
  '[Ar] 3d5 4s1',
  '[Ar] 3d5 4s2',
  '[Ar] 3d6 4s2',
  '[Ar] 3d7 4s2',
  '[Ar] 3d8 4s2',
  '[Ar] 3d10 4s1',
  '[Ar] 3d10 4s2',
  '[Ar] 3d10 4s2 4p1',
  '[Ar] 3d10 4s2 4p2',
  '[Ar] 3d10 4s2 4p3',
  '[Ar] 3d10 4s2 4p4',
  '[Ar] 3d10 4s2 4p5',
  '[Ar] 3d10 4s2 4p6',
  '[Kr] 5s1',
  '[Kr] 5s2',
  '[Kr] 4d1 5s2',
  '[Kr] 4d2 5s2',
  '[Kr] 4d4 5s1',
  '[Kr] 4d5 5s1',
  '[Kr] 4d5 5s2',
  '[Kr] 4d7 5s1',
  '[Kr] 4d8 5s1',
  '[Kr] 4d10',
  '[Kr] 4d10 5s1',
  '[Kr] 4d10 5s2',
  '[Kr] 4d10 5s2 5p1',
  '[Kr] 4d10 5s2 5p2',
  '[Kr] 4d10 5s2 5p3',
  '[Kr] 4d10 5s2 5p4',
  '[Kr] 4d10 5s2 5p5',
  '[Kr] 4d10 5s2 5p6',
  '[Xe] 6s1',
  '[Xe] 6s2',
  '[Xe] 5d1 6s2',
  '[Xe] 4f1 5d1 6s2',
  '[Xe] 4f3 6s2',
  '[Xe] 4f4 6s2',
  '[Xe] 4f5 6s2',
  '[Xe] 4f6 6s2',
  '[Xe] 4f7 6s2',
  '[Xe] 4f7 5d1 6s2',
  '[Xe] 4f9 6s2',
  '[Xe] 4f10 6s2',
  '[Xe] 4f11 6s2',
  '[Xe] 4f12 6s2',
  '[Xe] 4f13 6s2',
  '[Xe] 4f14 6s2',
  '[Xe] 4f14 5d1 6s2',
  '[Xe] 4f14 5d2 6s2',
  '[Xe] 4f14 5d3 6s2',
  '[Xe] 4f14 5d4 6s2',
  '[Xe] 4f14 5d5 6s2',
  '[Xe] 4f14 5d6 6s2',
  '[Xe] 4f14 5d7 6s2',
  '[Xe] 4f14 5d9 6s1',
  '[Xe] 4f14 5d10 6s1',
  '[Xe] 4f14 5d10 6s2',
  '[Xe] 4f14 5d10 6s2 6p1',
  '[Xe] 4f14 5d10 6s2 6p2',
  '[Xe] 4f14 5d10 6s2 6p3',
  '[Xe] 4f14 5d10 6s2 6p4',
  '[Xe] 4f14 5d10 6s2 6p5',
  '[Xe] 4f14 5d10 6s2 6p6',
  '[Rn] 7s1',
  '[Rn] 7s2',
  '[Rn] 6d1 7s2',
  '[Rn] 6d2 7s2',
  '[Rn] 5f2 6d1 7s2 ',
  '[Rn] 5f3 6d1 7s2',
  '[Rn] 5f4 6d1 7s2',
  '[Rn] 5f6 7s2',
  '[Rn] 5f7 7s2',
]

# cavalent radii for logaritmic derivative evaluation, from fhi98pp
_atom_rcov = [
  0.60, 1.76, 2.33, 1.70, 1.55, 1.46, 1.42, 1.38, 1.36, 1.34,
  2.91, 2.57, 2.23, 2.10, 2.00, 1.93, 1.87, 1.85, 3.84, 3.29,
  2.72, 2.50, 2.31, 2.23, 2.21, 2.21, 2.19, 2.17, 2.21, 2.36,
  2.38, 2.31, 2.27, 2.19, 2.16, 2.12, 4.08, 3.61, 3.06, 2.74,
  2.53, 2.46, 2.40, 2.36, 2.36, 2.42, 2.53, 2.80, 2.72, 2.67,
  2.65, 2.57, 2.51, 2.48, 4.44, 3.74, 3.19, 3.12, 3.12, 3.10,
  3.08, 3.06, 3.50, 3.04, 3.01, 3.01, 2.99, 2.97, 2.95, 2.95,
  2.95, 2.72, 2.53, 2.46, 2.42, 2.38, 2.40, 2.46, 2.53, 2.82,
  2.80, 2.78, 2.76, 2.76, 2.74, 2.71, 4.82, 3.88, 3.31, 3.12,
  3.00, 3.00, 3.00, 3.00, 3.00, 3.00, 3.00, 3.00, 3.00, 3.00,
]

if __name__ == '__main__':
    for z in range(1, 120):
        el = Element(z)
        config = parse_configuration(el.configuration)
        nelec = 0
        for orbital in config:
            nelec += orbital[2]
        print(el.symbol, el.name, el.rcov, z, nelec)
