#!/usr/bin/env python3
import freetype as ft
import numpy as np
import matplotlib.pyplot as plt
import bezier
import sys

face = ft.Face('./fonts/Arial Black.ttf')
face.set_char_size(48*64)
face.load_char(sys.argv[1])
outline = face.glyph.outline
print(outline.flags)

points = np.array(outline.points, dtype=np.double).transpose()
print('points.shape',np.array(outline.points).shape)
print('points\n',points)
tags = outline.tags
print('tags\n',tags)
curves = []

# loop through all points
# tags specify if point is on or off curve
start, end = 0,0
while end<len(tags):
	end=start+1
	while end<(len(tags)-1) and tags[end]&(1<<0)==0:
		end+=1
	# print('points[%d:%d] ='%(start,end),points[:,start:end+1])
	pad_len = 4-(end+1-start)

	# padding every curve to have 4 pts. Lines padded with last coordinate
	# and remain lines
	nodes = np.pad(points[:,start:end+1],((0,0),(0,pad_len)),'edge')	

	# nodes = points[:,start:end+1]
	curves.append(nodes)
	start=end

print('number of curves = %d'%len(curves))
for i,t in enumerate(tags):
	if t&(1<<0):
		print('tags[%s]=%s=%s'%('{0:02d}'.format(i),'{0:02d}'.format(t),'{0:07b}'.format(t)))
	
ax = plt.subplot()
for c in curves:
	curve = bezier.Curve.from_nodes(c)
	_ = curve.plot(num_pts=256,ax=ax)
for i,pt in enumerate(outline.points):
	ax.annotate(i,pt)
plt.show()

