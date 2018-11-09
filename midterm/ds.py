#!/usr/bin/python3
import scipy as sp
import scipy.misc
import imageio
import rawpy
import numpy as np
import matplotlib.pyplot as plt
import os, glob

base_dir = './'
in_dir   = base_dir + 'Sony/short/'
gt_dir   = base_dir + 'gt/'
val_dir  = 'gt/'
train_list = base_dir + 'Sony_train_list_png.txt'

gt_fns = glob.glob( gt_dir+'0*' )
train_ids = [int(os.path.basename(fn)[0:5]) for fn in gt_fns]
in_fns = [glob.glob(in_dir + '%05d_00*'%train_id)[0] for train_id in train_ids]

val_gt_fns = glob.glob(val_dir+'2*')
val_ids = [int(os.path.basename(fn)[0:5]) for fn in val_gt_fns]
print(val_ids)
print(glob.glob('Sony/short/'+str(val_ids[0])+'*'))
val_in_fns = [glob.glob('Sony/short/' + '%05d_*'%val_id)[0] for val_id in val_ids]
print(val_in_fns)

def mosaic2pk(mosaic):
	H = mosaic.shape[0]
	W = mosaic.shape[1]
	pk = np.dstack((mosaic[0:H:2,0:W:2], # R
				    mosaic[1:H:2,0:W:2], # G
				    mosaic[1:H:2,1:W:2], # B
				    mosaic[0:H:2,1:W:2]))# G
	return pk

def downsamp(raw,factor):
	img = raw.raw_image_visible.astype(np.float32)
	H = img.shape[0]
	dh = int(H/factor)
	W = img.shape[1]
	dw = int(W/factor)
	dwn = np.empty((dh,dw),dtype=np.float32)
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
		name = 'DWN%d/short/'%fac + os.path.basename(filename)[:-4]
		np.save(name,pk)
	
def rgb_downsamp_save(filename,factors):
	rgb = imageio.imread(filename)
	for fac in factors:
		dwn = scipy.misc.imresize(rgb,1/fac)
		name = 'DWN%d/long/'%fac + os.path.basename(filename)[:-4]
		scipy.misc.imsave(name+'.png',dwn)

rgb_downsamp_save(val_gt_fns[0],[16])
raw_downsamp_save(val_in_fns[0],[16])

in_fns = glob.glob('DWN16/short/2*')
gt_fns = glob.glob('DWN16/long/2*')

pk = np.load(in_fns[0])
print(pk.shape)
gt = scipy.misc.imread(gt_fns[0])
print(gt.shape)


figure, (ax1,ax2) = plt.subplots(1,2)
ax1.imshow(pk[:,:,0]/2**16)
ax2.imshow(gt)
plt.show()

for fn in in_fns:
	raw_downsamp_save(fn,[16])
	print(fn)
