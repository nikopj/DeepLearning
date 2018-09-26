# figure generation from tensorflow logs
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt

acc = genfromtxt('./tf_logs/run_.-tag-accuracy.csv', delimiter=',')
loss = genfromtxt('./tf_logs/run_.-tag-loss.csv', delimiter=',')

figure, (ax1,ax2) = plt.subplots(1,2)
ax1.plot(loss[:,2])
ax1.set_title("Loss")
ax1.set_xlabel("iterations (batches)")

ax2.plot(acc[:,2])
ax2.set_title("Accuracy of validation set")
ax2.set_xlabel("iterations (batches)")

plt.show()
