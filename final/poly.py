#!/usr/bin/python3

#Generates Regular Polygons and saves the data

from skimage.draw import polygon_perimeter
import matplotlib.pyplot as plt
import numpy as np

#Save Destination
filename = "./RegularPolyImgs"

#Data Set Parameters
perType = 500			#Number of data points per type
grid_size = 128			#Size of coordinate system
num_sides = 10			#Maximum amount of sides for generated polygons
showPoly = 5			#Amount to display

#Set up
shapeSides = np.arange(3,num_sides+1)					#Number of sides per shape
data = np.zeros((np.size(shapeSides)*perType, grid_size, grid_size),\
	dtype = np.uint8)	#Store data in here

#Data Generation
for i in range(np.size(shapeSides)):
	#5 in rand uniform to get distinguishable shapes
	radius = np.random.uniform(5,grid_size/2,perType)		#Random radius for each sample
	thetas = np.linspace(0,2*np.pi,shapeSides[i]+1)			#Linear spacing of theta (Regular Poly)

	for j in range(500):
		thetas += np.random.uniform(0,2*np.pi)				#Random phase (rotation)

		#Calculate vertices
		points = np.empty((2,shapeSides[i]+1))
		points[0,:] = (radius[j]*np.cos(thetas) + grid_size/2)
		points[0,:] = [int(pt) for pt in points[0,:]]				#Convert to int
		points[1,:] = (radius[j]*np.sin(thetas) + grid_size/2)
		points[1,:] = [int(pt) for pt in points[1,:]]				#Convert to int

		rows, columns = polygon_perimeter(points[0,:], points[1,:],\
			shape = [grid_size, grid_size], clip = True)

		data[i*perType + j,rows,columns] = 1		#Image

np.save(filename,data)			#Save Generated Data

#Enjoy the beauty!
for i in range(showPoly):
	ind = np.random.randint(np.shape(data)[0])
	img = data[ind,:,:]

	plt.figure(i)
	plt.imshow(img)

plt.show()
