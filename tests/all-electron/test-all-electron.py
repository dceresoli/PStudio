#!/usr/bin/env python
# coding: utf-8

# ## All electron calculations, comparing to the ld1.x code of Quamtum Espresso

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import subprocess

# add pstudio to the search path
import sys
sys.path.append('..')


# In[2]:


from pstudio import AE, set_output
from pstudio.configuration import *


# In[3]:


def ld1_create_input(atom, xcname='pz'):
    el = Element(atom)
    z = el.get_atomic_number()
    conf = el.get_configuration()
    
    ld1_in = """&input
        title = '{0}'
        prefix = '{0}'
        zed = {1}
        dft = '{2}'
        config = '{3}'
        rel = 1
        iswitch = 1
        beta = 0.2
        xmin = -8.0, dx = 0.005
        /""".format(atom, z, xcname, conf)
    return ld1_in

def ld1_run(inp, ld1='/home/ceresoli/Codes/q-e/bin/ld1.x'):
    p = subprocess.Popen(ld1, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    p.stdin.write(bytes(inp, encoding='ascii'))
    out = p.communicate()[0]
    p.stdin.close()
    return str(out, encoding='ascii')

def ld1_get_etot(out):
    pos1 = out.find('Etot') + 6
    pos2 = out.find(',', pos1) - 3
    etot = float(out[pos1:pos2])
    return etot/2.0 # rydberg to hartree


# In[4]:


def pstudio_etot(atom, xcname='lda'):
    set_output(None)
    ae = AE(atom, xcname)
    ae.run()
    return ae.Etot
    
def ld1_etot(atom, xcname='pz'):
    inp = ld1_create_input(atom, xcname)
    out = ld1_run(inp)
    return ld1_get_etot(out)


# In[8]:


print('=====================================================================')  
print('Atom          PStudio                LD1           Abs.err.  Rel.err.')  
print('=====================================================================')  
for atom in atom_table:
    pst = pstudio_etot(atom)
    ld1 = ld1_etot(atom)
    aerr = abs(pst-ld1)
    rerr = abs(pst-ld1)/abs(ld1) * 100
    print('{0:2s} {1:18.6f} {2:18.6f} {3:18.6f} {4:8.4f}%'.format(atom, pst, ld1, aerr, rerr))

