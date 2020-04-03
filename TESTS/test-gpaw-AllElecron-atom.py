#!/usr/bin/env python
# coding: utf-8

# In[1]:


from gpaw.atom.all_electron import AllElectron
import numpy as np
import matplotlib.pyplot as plt


# In[2]:

ti = AllElectron('Ti', xcname='LDA', scalarrel=True)
ti.run()


# In[4]:


fig = plt.figure(figsize=(10,7))
for i in range(len(ti.n_j)):
    plt.plot(ti.rgd.r_g, ti.u_j[i,:], label='n=%i, l=%i' % (ti.n_j[i], ti.l_j[i]))
plt.legend()
plt.xlim(0,5)
plt.show()


# In[5]:


rgd = ti.rgd
phi, c0 = rgd.pseudize(ti.u_j[6], gc=rgd.ceil(3.0), l=0)


# In[6]:


plt.plot(ti.rgd.r_g, phi)
plt.plot(ti.rgd.r_g, ti.u_j[6])
plt.xlim(0,5)


# In[ ]:




