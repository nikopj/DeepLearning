#!/usr/bin/python3
import scipy as sp
import scipy.misc
import pandas
import imageio
import rawpy
import numpy as np
import matplotlib.pyplot as plt
import os

base_dir = './'
in_dir   = base_dir + 'Sony/short/'
gt_dir   = base_dir + 'gt/'
train_list = base_dir + 'Sony_train_list_png.txt'

df = pandas.read_csv(train_list, sep=' ', index_col=False, 
	header=None, names=['in','gt','iso','fstop'], lineterminator='\n')

s = np.arange(len(df['in']))
# np.random.shuffle(s)
pn_xtrn = base_dir + np.array(df['in'])[s]
pn_ytrn = base_dir + np.array(df['gt'])[s]

ids = []
ratios = []
xi = pn_xtrn[0].rfind('/')
yi = pn_ytrn[0].rfind('/')
for i in range(len(pn_xtrn)):
	ids.append(pn_xtrn[i][xi+2:xi+6])
	ratios.append(float(pn_ytrn[i][yi+10:-5]) / float(pn_xtrn[i][xi+10:-5]))

def mosaic2pk(mosaic):
	H = mosaic.shape[0]
	W = mosaic.shape[1]
	pk = np.dstack((mosaic[0:H:2,0:W:2], # R
				    mosaic[1:H:2,0:W:2], # G
				    mosaic[1:H:2,1:W:2], # B
				    mosaic[0:H:2,1:W:2]))# G
	return pk

def downsamp(raw,factor):
	img = raw.raw_image_visible.astype(np.int16)
	H = img.shape[0]
	dh = int(H/factor)
	W = img.shape[1]
	dw = int(W/factor)
	dwn = np.empty((dh,dw),dtype=np.int16)
	dwn[0:dh:2,0:dw:2]=img[0:H:2*factor,0:W:2*factor]
	dwn[1:dh:2,0:dw:2]=img[1:H:2*factor,0:W:2*factor]
	dwn[1:dh:2,1:dw:2]=img[1:H:2*factor,1:W:2*factor]
	dwn[0:dh:2,1:dw:2]=img[0:H:2*factor,1:W:2*factor]
	return dwn

def raw_downsamp_save(filename,factors):
	raw = rawpy.imread(filename)
	for fac in factors:
		dwn = downsamp(raw,fac)
		pk  = mosaic2pk(dwn)
		name = base_dir+'DWN%d/short/'%fac + os.path.basename(filename[:-4])
		scipy.misc.imsave(name+'.png',dwn)
		# np.save(name,pk)
	
def rgb_downsamp_save(filename,factors):
	rgb = imageio.imread(filename)
	for fac in factors:
		dwn = scipy.misc.imresize(rgb,1/fac)
		name = base_dir+'DWN%d/long/'%fac + os.path.basename(filename[:-4])
		scipy.misc.imsave(name+'.png',dwn)
		# np.save(name,dwn)

rgb_downsamp_save(pn_ytrn[0],[4])
raw_downsamp_save(pn_xtrn[0],[4])

pk = scipy.misc.imread('./DWN4/short/'+os.path.basename(pn_xtrn[0]))
print(pk.shape)
gt = scipy.misc.imread('./DWN4/long/'+os.path.basename(pn_ytrn[0]))
print(gt.shape)



# for fn in pn_xtrn:
# 	raw_downsamp_save(fn,[4,8])
# for fn in pn_ytrn
# 	rgb_downsamp_save(fn,[4,8])
