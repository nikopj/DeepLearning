#!/usr/bin/python3
import scipy as sp
import scipy.misc
import pandas
import imageio
import rawpy
import numpy as np
import matplotlib.pyplot as plt

base_dir = '/home/nikopj/Documents/SONY/'
in_dir   = base_dir + 'Sony/short/'
gt_dir   = base_dir + 'gt/'
train_list = base_dir + 'Sony_train_list_png.txt'

df = pandas.read_csv(train_list, sep=' ', index_col=False, 
	header=None, names=['in','gt','iso','fstop'], lineterminator='\n')

s = np.arange(len(df['in']))
np.random.shuffle(s)
pn_xtrn = base_dir + np.array(df['in'])[s]
pn_ytrn = base_dir + np.array(df['gt'])[s]

ids = []
ratios = []
xi = pn_xtrn[0].rfind('/')
yi = pn_ytrn[0].rfind('/')
for i in range(len(pn_xtrn)):
	ids.append(pn_xtrn[i][xi+2:xi+6])
	ratios.append(float(pn_ytrn[i][yi+10:-5]) / float(pn_xtrn[i][xi+10:-5]))
	
print(ids[10],pn_ytrn[10],pn_xtrn[10],ratios[10])

i = 10
raw = rawpy.imread(pn_xtrn[i])
gt = imageio.imread(pn_ytrn[i])

# from SID train_Sony.py line 164
# rgb = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=bps)

# --------------------------------------------
# ------------------- DATA -------------------
# --------------------------------------------

# BAYER ARRAY to 4 Channel Pack
# RG
# GB to [R,G,B,G]
def byr2pk(raw):
	img = raw.raw_image_visible.astype(np.float32)
	# subtrack black level ?
	# raw.black_level_per_channel shows that the black level is 512
	# subtract 512 from each pixel (or clip at 0) and put the values
	# on a range [0,1] by dividing by (2**14 -1 - 512)
	# (as the raw files are 14bit precision)
	# 2**14 == 16384
	img = np.maximum(img-512,0) / (16383-512)

	H = img.shape[0]
	W = img.shape[1]
	pk = np.dstack((img[0:H:2,0:W:2], # R
				    img[1:H:2,0:W:2], # G
				    img[1:H:2,1:W:2], # B
				    img[0:H:2,1:W:2]))# G
	# put pack in batch ready form
	pk = np.expand_dims(pk,0)
	return pk

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
	img = np.maximum(img-512,0) / (16383-512)
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

# pk: input image in pack form
# gt: ground truth image (in sRGB)
# ps: patch size (square)
# a patch of the packed bayers corresponds to a patch
# of the full size image (gt) that is twice as large
def to_patch(pk,gt,ps=1024):
	shp = pk.shape
	H = pk.shape[1]
	W = pk.shape[2]
	h = np.random.randint(H-ps)
	w = np.random.randint(W-ps)
	pk_patch = pk[:, h:(h+ps),     w:(w+ps),    :]
	gt_patch = gt[:, 2*h:2*(h+ps), 2*w:2*(w+ps),:]
	return pk_patch, gt_patch

# im = raw.raw_image_visible
# print(im.shape) 

# pk = byr2pk(raw)
dwn = downsamp(raw,8)
pk = mosaic2pk(dwn)
np.save('test1',pk)
lpk = np.load('test1.npy')

dwngt = scipy.misc.imresize(gt,1/8)
np.save('gttest',dwngt)
ldwngt = np.load('gttest.npy')
# pk_patch, gt_patch = to_patch(pk,gt)

# demo
figure, (ax1,ax2) = plt.subplots(1,2)
ax1.imshow(np.squeeze(lpk[:,:,0]))
ax2.imshow(ldwngt)
plt.show()

# https://github.com/letmaik/rawpy/blob/master/test/basic_tests.py
# print('black_level_per_channel:', raw.black_level_per_channel)
# print('color_matrix:', raw.color_matrix)
# print('rgb_xyz_matrix:', raw.rgb_xyz_matrix)
# print('tone_curve:', raw.tone_curve)

