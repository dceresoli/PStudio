# encoding: utf-8
# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

"""Calculates radial Coulomb integral"""

from math import pi

def hartree(l, nrdr, r, vr):
    """Calculates radial Coulomb integral.

    The following integral is calculated::

                                   ^
                          n (r')Y (r')
              ^    / _     l     lm
      v (r)Y (r) = |dr' --------------,
       l    lm     /        _   _
                           |r - r'|

    where input and output arrays `nrdr` and `vr`::

              dr
      n (r) r --  and  v (r) r.
       l      dg        l
    """
    assert nrdr.shape == vr.shape and len(vr.shape) == 1
    assert len(r.shape) == 1
    assert len(r) >= len(vr)

    M = nrdr.shape[0]
    p = 0.0
    q = 0.0

    for g in range(M-1, 0, -1):
        R = r[g]
        rl = R**l
        dp = nrdr[g] / rl
        rlp1 = rl * R
        dq = nrdr[g] * rlp1
        vr[g] = (p + 0.5*dp) * rlp1 - (q + 0.5*dq)/rl
        p += dp
        q += dq

    vr[0] = 0.0
    f = 4.0*pi / (2*l+1)

    for g in range(1, M):
        R = r[g]
        vr[g] = f* (vr[g] + q / R**l)

    return None
