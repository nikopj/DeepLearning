#!/usr/bin/env python3
import freetype as ft
from freetype.ft_enums.ft_curve_tags import FT_CURVE_TAG
import numpy as np
import matplotlib.pyplot as plt
import bezier
import sys

DEBUG = 0

CONIC = 0
ON    = 1
CUBIC = 2

face = ft.Face('./fonts/Georgia Bold.ttf')
face.set_char_size(48*64)
face.load_char(sys.argv[1])
outline = face.glyph.outline

# points (2,n_pts)
points = np.array(outline.points, dtype=np.double).transpose()
# tags, cubic, conic, or on
tags = outline.tags
curves = []

if DEBUG:
	print('tags\n',tags)

j = 0 # pt indices
# iterate over the number of contours 
for i in range(outline.n_contours):
	# starting point is just past the end of the previous contour
	if i==0:
		start=0
	else:
		start=outline.contours[i-1]+1	
	j=start	
	end = outline.contours[i]

	# iterate over all points in contour
	while j<=end: 
		node_indices = []
		on_count=0
		conic_count=0
		cubic_count=0
		while on_count<2 and j<=end:
			# last point is first point	
			if j==start and FT_CURVE_TAG(tags[start])!=ON and FT_CURVE_TAG(tags[end])==ON:
				node_indices.append(end)
				on_count+=1
			
			# adding node_indices points
			if FT_CURVE_TAG(tags[j])==ON:
				node_indices.append(j)
				on_count+=1
			if FT_CURVE_TAG(tags[j])==CONIC:
				node_indices.append(j)
				conic_count+=1
			if FT_CURVE_TAG(tags[j])==CUBIC:
				node_indices.append(j)
				cubic_count+=1

			# first point is last point
			if j==end and FT_CURVE_TAG(tags[start])==ON and FT_CURVE_TAG(tags[end])!=ON:
				node_indices.append(start)
				on_count+=1
			# first and last point are off --> first and last point are the midpoint
			# add to the count to get out the loop, check again once outside
			if (j==start or j==end) and FT_CURVE_TAG(tags[start])!=ON and FT_CURVE_TAG(tags[end])!=ON:
				on_count+=1
			# allows end point of one curve to be start point of next curve
			if on_count<2:
				j+=1
		if j>end:
			break
	
		# nodes numpy array (2,num_pts)
		nodes = points[:,node_indices]
			
		if (j==start or j==end) and FT_CURVE_TAG(tags[start])!=ON and FT_CURVE_TAG(tags[end])!=ON:
			midpoint = np.reshape((points[:,start]+points[:,end])/2,(2,1))
			if j==start:
				nodes = np.insert(nodes,0,midpoint,axis=1)
			if j==end:
				nodes = np.concatenate((nodes,midpoint),axis=1)
		if cubic_count==1:
			print('too few cubic points for node_indices',nodes,file=sys.stderr)
		if conic_count>1:
			k=1
			mps = []
			r=1
			while r<conic_count:
				r+=1
				midpoint = np.reshape((nodes[:,k]+nodes[:,k+1])/2,(2,1))
				midpoint = np.repeat(midpoint,2,axis=1)
				nodes = np.insert(nodes,k+1,midpoint.transpose(),axis=1)
				mps.append(k+2)
				k+=3

			split_nodes = np.split(nodes,mps,axis=1)
			curves.extend(split_nodes)	

		# appending complete bezier curve
		else:
			curves.append(nodes)

	if FT_CURVE_TAG(tags[start])==ON and FT_CURVE_TAG(tags[end])==ON:
		nodes = points[:,(start,end)]
		curves.append(nodes)

ax = plt.subplot()
for c in curves:
	curve = bezier.Curve.from_nodes(c)
	_ = curve.plot(num_pts=256,ax=ax)
	plt.draw()
	plt.pause(0.0001)

for i,pt in enumerate(outline.points):
	ax.annotate(i,pt)
plt.show()
