#!/usr/bin/python3

import matplotlib.pyplot as plt
import numpy as np
from skimage.draw import polygon_perimeter

grid_size = 4096;

n = np.random.randint(4,11)

radius = np.random.uniform(0,grid_size/2)
thetas = np.linspace(0,2*np.pi,n) + np.random.uniform(0,2*np.pi)

nodes = np.empty((2,n))
for i in range(n):
	nodes[0,i] = int(radius*np.cos(thetas[i]) + grid_size/2)
	nodes[1,i] = int(radius*np.sin(thetas[i]) + grid_size/2)

print(nodes,thetas,radius)

img = np.zeros((grid_size,grid_size), dtype=np.uint8)
rr,cc = polygon_perimeter(nodes[0,:],nodes[1,:],shape=img.shape,clip=True)
img[rr,cc] = 1

plt.imshow(img)
plt.show()
