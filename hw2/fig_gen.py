# figure generation
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import *

nsamp = 300
var = .1
# noise
v = np.sqrt(var/2)*randn(nsamp,4)

# class 1
t1 = uniform(np.pi/4, 4*np.pi, nsamp)
x1 = np.array([ t1*np.sin(t1)+v[:,0], t1*np.cos(t1)+v[:,1] ])

# class 2
t2 = uniform(np.pi/4, 4*np.pi, nsamp)
x2 = np.array([ -t2*np.sin(t2)+v[:,2], -t2*np.cos(t2)+v[:,3] ])

plt.figure
plt.plot(x1[0,:], x1[1,:],
    'ob', markeredgecolor="black")
plt.plot(x2[0,:], x2[1,:],
    'or', markeredgecolor="black")
plt.show()
