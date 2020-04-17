"""
Exchange and correlation functional
"""

has_libxc = True
try:
    from .pylibxc import LibXCFunctional
    from .pylibxc.util import *
except ImportError:
    has_libxc = False

import numpy as np

known_functionals = {
  'lda'   : 'lda_x+lda_c_pz',
  'pz'    : 'lda_x+lda_c_pz',
  'pw91'  : 'gga_x_pw91+gga_c_pw91',
  'blyp'  : 'gga_x_b88+gga_c_lyp',
  'pbe'   : 'gga_x_pbe+gga_c_pbe',
  'pbesol': 'gga_x_pbe_sol+gga_c_pbe_sol',
}

class XC:
    """Wrap libxc or provide PZ-LDA implemented in Python"""
    def __init__(self, xcname, spin=1):
        xcname = xcname.lower()
        if not has_libxc or xcname == 'lda-py':
            if xcname != 'lda-py' and spin != 1:
                raise RuntimeError('only LDA is available without libxc')
            self.xcname = xcname
            self.xc = None
            self.spin = spin
            return

        # abbreviations
        if xcname in known_functionals:
            xcname = known_functionals[xcname]       # expand definition

        self.xcname = xcname
        self.spin = spin
        self.xc = []

        xcnames = [x.strip() for x in xcname.split('+')]
        for name in xcnames:
            try:
                num = int(name)
            except:
                num = xc_functional_get_number(name)
            self.xc.append(LibXCFunctional(num, spin=spin))

    def get_description(self):
        if self.xc == None:
            return "LDA (PZ parametrization) in python"

        desc = '\n\n'.join([xc.describe() for xc in self.xc])
        return desc

    def get_name(self):
        if self.xc == None:
            return "LDA-python"

        name = '+'.join([xc._xc_func_name for xc in self.xc])
        return name

    def compute(self, rho, sigma):
        if self.xc == None:
            exc = np.zeros_like(rho)
            vxc = np.zeros_like(rho)
            for i in range(len(rho)):
                exc[i], vxc[i] = lda_xc(rho[i])
            return exc, vxc

        inp = {'rho': rho, 'sigma': sigma}
        out = self.xc[0].compute(inp)
        for i in range(1, len(self.xc)):
            tmp = self.xc[i].compute(inp)
            for k in tmp.keys():
                out[k] += tmp[k]

        # the reshaping is to kill the first dimension
        if 'vsigma' not in out:
            out['vsigma'] = np.zeros_like(rho)

        return out['zk'].reshape((-1)), \
               out['vrho'].reshape((-1)), \
               out['vsigma'].reshape((-1))



# pure python implementation of LDA PZ
from math import log, exp, sqrt

pi34 = 0.6203504908994  # (3.0/(4.0*pi))**(1.0/3.0)
third = 1.0/3.0

def lda_xc(rho):
    """LDA PZ in python"""
    if abs(rho) < 1e-14:
        return 0.0, 0.0

    rs = pi34 / rho**third

    # Slater exchange
    f = -0.687247939924714  # -(9.0/8.0)*(3.0/(2.0*pi))**(2.0/3.0)
    alpha = 2.0/3.0

    ex = f * alpha / rs
    vx = 4.0/3.0 * f * alpha / rs

    # PZ correlation
    a, b, c, d = 0.0311, -0.048, 0.002, -0.0116
    gc, b1, b2 = -0.1423, 1.0529, 0.3334
    if rs < 1.0:
        # high density formula
        lnrs = log(rs)
        ec = a*lnrs + b + c*rs*lnrs + d*rs
        vc = a*lnrs + (b-a/3.0) + (2.0/3.0) * c*rs*lnrs + (2.0*d-c)/3.0*rs
    else:
        # interpolation formula
        rs12 = sqrt(rs)
        ox = 1.0 + b1*rs12 + b2*rs
        dox = 1.0 + (7.0/6.0)*b1*rs12 + 4.0/3.0*b2*rs
        ec = gc / ox
        vc = ec * dox / ox

    return ex + ec, vx + vc   # in Hartree
