#!/usr/bin/python3
import scipy as sp
import matplotlib.pyplot as plt
import pandas
import imageio
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import rawpy
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

logs_dir = './logs/'
tflog = logs_dir + 'tflog'
logfile = logs_dir + 'log.txt'
check_point_dir = './ckpnt/'
base_dir = '/home/nikopj/Documents/SONY/'
in_dir   = base_dir + 'Sony/short/'
gt_dir   = base_dir + 'gt/'
train_list = base_dir + 'Sony_train_list_png.txt'

# hyper-parameters
BATCH_SIZE = 1
NUM_EPOCHS = 1
SAVE_RATE = 5
LEARNING_RATE = 1e-4
PATCH_SIZE = 256

# -----------------------------------------------
# ------------------- NETWORK -------------------
# -----------------------------------------------

# activation fcn used in architecture
def act(x):
	return tf.nn.relu6(x)

def conv(x,chan,ksz):
	return tf.layers.conv2d(x,chan,ksz,padding='SAME',activation=act)

# contraction module of UNET
def contract_mod(x,chans=None):
	num_fchans = 2*int(x.shape[-1]) if chans==None else chans
	c = conv(x,num_fchans,3)
	c = conv(c,num_fchans,3)
	p = tf.layers.max_pooling2d(c,2,2,padding='SAME')
	return c,p

# upsample in1, concatenate both inputs
# output 2*c feature channels
def upsample_cat(in1,in2,c):
	ups = tf.layers.conv2d_transpose(in1,c,2,2,padding='SAME')
	x = tf.concat([ups,in2],3)
	return x

# expansion module of UNET
# input1 upsample connection, input2 concat connection
def expand_mod(input1,input2):
	out_chans= input2.get_shape()[-1]
	x = upsample_cat(input1,input2,out_chans)
	x = conv(x,out_chans,3)
	x = conv(x,out_chans,3)
	return x

# UNET ARCHITECTURE
def f(x):
	# contract
	c1,p1 = contract_mod(x,chans=16)
	c2,p2 = contract_mod(p1)
	c3,p3 = contract_mod(p2)
	c4,p4 = contract_mod(p3)
	c5, _ = contract_mod(p4)
	# expand
	e1 = expand_mod(c5,c4)
	e2 = expand_mod(e1,c3)
	e3 = expand_mod(e2,c2)
	e4 = expand_mod(e3,c1)
	# out of UNET
	sub = tf.layers.conv2d(e4,12,1,padding='SAME')	
	# depth2space == subpixel reconstruction
	# out chans =  12/(2*2) = 3 <-- perfect!
	out = tf.depth_to_space(sub,2)
	return out

# ---------------------------------------------
# ------------------- FILES -------------------
# ---------------------------------------------

# load train txt file into data frame
df = pandas.read_csv(train_list, sep=' ', index_col=False, 
	header=None, names=['in','gt','iso','fstop'], lineterminator='\n')

# shuffle pathnames
s = np.arange(len(df['in']))
np.random.shuffle(s)
pn_xtrn = base_dir + np.array(df['in'])[s]
pn_ytrn = base_dir + np.array(df['gt'])[s]

# get id number and exposure ratio of input ground truth pair
ids = []
ratios = []
xi = pn_xtrn[0].rfind('/')
yi = pn_ytrn[0].rfind('/')
for i in range(len(pn_xtrn)):
	ids.append(pn_xtrn[i][xi+2:xi+6])
	ratios.append(float(pn_ytrn[i][yi+10:-5]) / float(pn_xtrn[i][xi+10:-5]))

in_imgs = {}
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
	return pk

# pk: input image in pack form
# gt: ground truth image (in sRGB)
# ps: patch size (square)
# a patch of the packed bayers corresponds to a patch
# of the full size image (gt) that is twice as large
def to_patch(pk,gt,ps=PATCH_SIZE):
	shp = pk.shape
	H = pk.shape[0]
	W = pk.shape[1]
	h = np.random.randint(H-ps)
	w = np.random.randint(W-ps)
	pk_patch = pk[h:(h+ps),     w:(w+ps),    :]
	gt_patch = gt[2*h:2*(h+ps), 2*w:2*(w+ps),:]
	return pk_patch, gt_patch

def getxy(pn_x,pn_y):
	pk = byr2pk(rawpy.imread(pn_x))
	gt = imageio.imread(pn_y)
	return to_patch(pk,gt)

def get_batch():
	l = len(pn_xtrn)
	s = np.random.randint(l,size=BATCH_SIZE)
	xb=[]
	yb=[]
	for i in s:
		pkp,gtp = getxy(pn_xtrn[i],pn_ytrn[i])
		pkp = pkp*ratios[i]
		xb.append(pkp), yb.append(gtp)
	return np.stack(xb), np.stack(yb)

# demo of batches
# xb,yb = get_batch()
# print(xb.shape,yb.shape)
# figure, (ax1,ax2) = plt.subplots(1,2)
# ax1.imshow(np.squeeze(xb[:,:,:,0]))
# ax2.imshow(np.squeeze(yb))
# plt.show()

# ------------------------------------------------
# ------------------- TRAINING -------------------
# ------------------------------------------------

# learning rate
lr = tf.placeholder(tf.float32)
# input, truth, est.
x = tf.placeholder(tf.float32, shape=[None,None,None,4])
y = tf.placeholder(tf.float32, shape=[None,None,None,3])
y_hat = f(x)

# L1 LOSS ONLY
loss = tf.reduce_mean( tf.abs(y_hat - y) )
optim = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)
init  = tf.global_variables_initializer()

# Create a summary to monitor cost tensor
tf.summary.scalar("loss", loss)
# Create a summary to monitor accuracy tensor
# tf.summary.scalar("accuracy", accuracy)
# Merge all summaries into a single op
merged_summary_op = tf.summary.merge_all()

with tf.Session() as sess:
	sess.run(init)

	# op to write logs to Tensorboard
	summary_writer = tf.summary.FileWriter(tflog,
		graph=tf.get_default_graph())
	learning_rate = 1e-4 
	# training
	for epoch in range(NUM_EPOCHS):
		avg_cost = 0.
		num_batches = int( np.ceil( len(pn_xtrn) / BATCH_SIZE ) )

		if epoch>2000:
			learning_rate = 1e-5

		for i in tqdm(range(num_batches)):
			xb, yb = get_batch()
			fd = {x: xb, y: yb, lr: learning_rate}
			output, loss_np, _, summary \
				= sess.run([y_hat, loss, optim, merged_summary_op], feed_dict=fd)
			# logs every batch
			summary_writer.add_summary(summary, epoch * num_batches + i)
			avg_cost  += loss_np/num_batches

			# Display logs per epoch step
			if (i+1) % SAVE_RATE == 0:
				logf = open(logfile,'a')
				description = 'Epoch %02d, Batch %04d\n'%(epoch+1,i+1)
				logf.write(description)
				logf.close()
				cat = np.concatenate((output[0,:,:,:],yb[0,:,:,:]),axis=1)
				sp.misc.imsave(logs_dir + '%02d%04d.jpg'%(epoch+1,i+1), cat)

		# print('Validation Set Accuracy:',
			# accuracy.eval({x: data.x_val, y: data.y_val, phase: False}))

	print("Run the command line:\n--> tensorboard --logdir=./tf_logs ")
