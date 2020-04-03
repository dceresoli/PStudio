def divrl(a_g, l, r_g):
    """Return array divided by r to the l'th power."""
    b_g = a_g.copy()
    if l > 0:
        b_g[1:] /= r_g[1:]**l
        b1, b2 = b_g[1:3]
        r12, r22 = r_g[1:3]**2
        b_g[0] = (b1 * r22 - b2 * r12) / (r22 - r12)
    return b_g

