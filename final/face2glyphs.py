#!/usr/bin/env python3
from freetype import *
from freetype.ft_enums.ft_curve_tags import FT_CURVE_TAG
import numpy as np
import matplotlib.pyplot as plt
import bezier
import sys
import string
import glob

# size of numpy array that stores glpyph bitmap
IM_SHAPE = (32,32)
# shape font file attempts to render glyph at
# can often be a bit larger than hoped for, so make IM_SHAPE larger
IM_RENDER = (32,32)

chars = string.ascii_letters + string.digits

def outline2bez(outline):
	CONIC = 0
	ON    = 1
	CUBIC = 2

	points = np.array(outline.points, dtype=np.double).transpose()
	tags = outline.tags
	curves = []

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

	# CONVERTING CURVE LIST TO 4xMx2 NUMPY ARRAY, 
	# WHERE M IS THE NUMBER OF CURVES IN THE GLYPH
	np_glyph = np.empty((4,len(curves),2),dtype=np.float32)
	for i in range(len(curves)):
		# elevate all curvers to 3rd order (4 pts)
		while curves[i].shape[1]<4:
			c = bezier.Curve.from_nodes(curves[i])
			curves[i] = c.elevate().nodes
		np_glyph[:,i,:] = curves[i].transpose()
	
	return np_glyph

# from freetype's glyph-monochrome.py
# simple int to 8bit vector
def bits(x):
	data = []
	for i in range(8):
		data.insert(0, int((x & 1) == 1))
		x = x >> 1
	return data

# from freetype's glyph-monochrome.py
# returns numpy array of glyph bitmap
def bitmap2im(bitmap):
	width  = bitmap.width
	rows   = bitmap.rows
	pitch  = bitmap.pitch

	data = []
	for i in range(rows):
		row = []
		for j in range(pitch):
			row.extend(bits(bitmap.buffer[i*pitch+j]))
		data.extend(row[:width])
	Z = np.array(data).reshape(rows, width)
	pad = np.array(IM_SHAPE)-np.array(Z.shape)
	pad = ( ( int(np.floor(pad[0]/2)), int(np.ceil(pad[0]/2)) ) , 
			( int(np.floor(pad[1]/2)), int(np.ceil(pad[1]/2)) ) )
	Z = np.pad(Z,pad,'constant',constant_values=0)
	return Z

def plot_glyph(np_glyph,ax,animate=False,annotate=False,points=None):
	for i in range(np_glyph.shape[1]):
		curve = bezier.Curve.from_nodes(np_glyph[:,i,:].transpose())
		_ = curve.plot(num_pts=256,ax=ax)
		if animate is True:
			plt.draw()
			plt.pause(0.01)
	if annotate is True and points is not None:
		for i,pt in enumerate(points):
			ax.annotate(i,pt)


filenames = glob.glob('./fonts/*tf')
i = np.random.randint(0,len(filenames))
face = Face(filenames[i])
face.set_char_size(48*64)
face.set_pixel_sizes(IM_RENDER[0],IM_RENDER[1])
face_glyphs = []
face_imgs = []
print(filenames[i])
for char in chars:
	print('... '+char)
	face.load_char(char, FT_LOAD_RENDER | FT_LOAD_TARGET_MONO)
	outline = face.glyph.outline
	bitmap = face.glyph.bitmap

	face_glyphs.append(outline2bez(outline))
	try:
		bm = bitmap2im(bitmap)
	except:
		print('font irregular, skipping')
		sys.exit()
	face_imgs.append(bm)


i = np.random.randint(0,len(chars))
fig, [ax1,ax2] = plt.subplots(1,2)
plot_glyph(face_glyphs[i],ax1)
ax2.imshow(face_imgs[i])
plt.show()
