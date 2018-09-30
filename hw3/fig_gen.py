# figure generation from tensorflow logs
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt

acc = genfromtxt('./tf_logs/run_.-tag-accuracy.csv', delimiter=',')
loss = genfromtxt('./tf_logs/run_.-tag-loss.csv', delimiter=',')

figure, (ax1,ax2) = plt.subplots(1,2)
ax1.semilogy(loss[:,2])
ax1.set_title("Loss")
ax1.set_xlabel("iterations (batches)")
ax1.set_xlim([0, 1000])
ax1.set_ylim([1e-1, 25])
ax2.set_yticks(np.arange(0,25))
ax1.grid(which='both')

ax2.plot(acc[:,2])
ax2.set_title("Accuracy of validation set")
ax2.set_xlabel("iterations (batches)")
ax2.set_xlim([0, 1000])
ax2.set_ylim([0, 1])
ax2.set_yticks(np.arange(0,1.1,.1))
ax2.set_yticks(np.arange(0.05,1,.05), minor=True)
#ax2.set_yticks(np.linspace(0,1000,20))
ax2.grid(which='both')

plt.show()
