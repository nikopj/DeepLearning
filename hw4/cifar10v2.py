import os
import pickle
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.datasets.cifar10 import load_data

logs_path = "./tf_logs/"

input_size = 32 # 32x32 imgs
num_channels = 3 # RGB
num_classes = 10

# hyper-parameters
BATCH_SIZE = 300
NUM_EPOCHS = 50
display_epoch = 1
LEARNING_RATE = 1
# regularization parameters
drop_prob = 0.25
reg_scale = 1e-6

# https://stackoverflow.com/questions/38592324/one-hot-encoding-using-numpy/38592416
def get_one_hot(targets, nb_classes):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape)+[nb_classes])

class Data(object):
    def __init__(self):
        np.random.seed(31415)
        (self.x_train, y_train), (x_test, y_test) = load_data()

        (self.x_train, x_test) = (self.x_train / 255.0, x_test / 255.0)

        [(self.x_val, self.x_test), (y_val, y_test)] = \
            [np.split(var,2) for var in [x_test, y_test]]

        [self.y_train, self.y_val, self.y_test] = \
            [np.squeeze(get_one_hot(y, num_classes)) for y in [y_train,y_val,y_test]]

        self.index = np.arange(self.x_train.shape[0])

    # CUTOUT Regularizaiton (code adapted from the cutout team's github)
    # https://github.com/uoguelph-mlrg/Cutout
    def cutout(self, img, n_holes, length):
        h = img.shape[0]
        w = img.shape[1]

        mask = np.ones((h, w), np.float32)

        for n in range(n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - length // 2, 0, h)
            y2 = np.clip(y + length // 2, 0, h)
            x1 = np.clip(x - length // 2, 0, w)
            x2 = np.clip(x + length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask3 = np.zeros((h,w,3))
        mask3[:,:,0] = mask
        mask3[:,:,1] = mask
        mask3[:,:,2] = mask

        return img*mask3

    def get_batch(self):
        choices = np.random.choice(self.index, size=BATCH_SIZE)
        x = self.x_train[choices,:,:,:]
        cutout = np.empty((BATCH_SIZE,32,32,3))
        for i in range(BATCH_SIZE):
            cutout[i,:,:,:] = self.cutout(x[i,:,:,:], 1, 10)

        return cutout, self.y_train[choices,:]


# -------------- MODEL FUNCTIONS -----------------

# -------------- MODEL FUNCTIONS -----------------

# --- CONV LAYER WRAPPER --- w/ L2 regularization
# conv -> dropout -> BN -> relu -> max_pool
def conv_layer(input, filters, kernel_size, strides=2, is_training=True):
    x = tf.layers.conv2d(
        input, filters, kernel_size, strides=strides, padding='same',
        kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=reg_scale)
    )
    return x

# --- FULLY CONNECTED LAYER WRAPPER ---
# matmul -> dropout -> BN -> relu
def fc_layer(input, units, is_training=True):
    x = tf.layers.dense(
        input, units,
        kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=reg_scale)
    )
    x = tf.layers.batch_normalization(x, training=is_training, renorm=True)
    x = tf.nn.relu6(x)
    x = tf.layers.dropout(x, rate=drop_prob, training=is_training)
    return x

x = tf.placeholder(tf.float32, [None,input_size,input_size,num_channels])
y = tf.placeholder(tf.float32, [None,num_classes])
phase = tf.placeholder(tf.bool) # is_training
lr = tf.placeholder(tf.float32) # is_training

# ---- NN -----

def f(x):
    x = conv_layer(x, 16, 5, strides=1, is_training=phase)
    print(x.get_shape())
    x = conv_layer(x, 16, 3, strides=1, is_training=phase)
    print(x.get_shape())

    x = tf.layers.batch_normalization(x, training=phase, renorm=True)
    x = tf.nn.relu6(x)

    x = tf.layers.max_pooling2d(x, 3, 2, padding='same')
    print(x.get_shape())

    x = tf.layers.dropout(x, rate=drop_prob, training=phase)

    x = conv_layer(x, 32, 3, is_training=phase)
    print(x.get_shape())
    x = conv_layer(x, 32, 3, is_training=phase)
    print(x.get_shape())

    x = tf.layers.batch_normalization(x, training=phase, renorm=True)
    x = tf.nn.relu6(x)

    x = tf.layers.average_pooling2d(x, 3, 2, padding='same')
    print(x.get_shape())

    x = conv_layer(x, 64, 3, is_training=phase)
    print(x.get_shape())
    x = conv_layer(x, 64, 3, is_training=phase)
    print(x.get_shape())

    x = tf.layers.batch_normalization(x, training=phase, renorm=True)
    x = tf.nn.relu6(x)

    x = tf.layers.average_pooling2d(x, 3, 2, padding='same')
    print(x.get_shape())

    x = tf.layers.dropout(x, rate=drop_prob, training=phase)

    x = tf.layers.flatten(x)
    x = fc_layer(x, 32, is_training=phase)
    x = tf.layers.dense(x, num_classes)
    return x

# models
logits = f(x)
prediction = tf.nn.softmax(logits)

correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# LOSS
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    loss = tf.reduce_mean( tf.losses.softmax_cross_entropy(y, logits) ) \
        + tf.reduce_mean( tf.losses.get_regularization_loss() )
    optim = tf.train.AdamOptimizer(learning_rate=.01).minimize(loss)

init = tf.global_variables_initializer()

# Create a summary to monitor cost tensor
tf.summary.scalar("loss", loss)
# Create a summary to monitor accuracy tensor
tf.summary.scalar("accuracy", accuracy)
# Merge all summaries into a single op
merged_summary_op = tf.summary.merge_all()

with tf.Session() as sess:
    data = Data()
    sess.run(init)
    # op to write logs to Tensorboard
    summary_writer = tf.summary.FileWriter(logs_path,
        graph=tf.get_default_graph())
    learning_rate = LEARNING_RATE
    # training
    for epoch in range(NUM_EPOCHS):
        avg_cost = 0.
        num_batches = int( np.ceil( data.index[-1] / BATCH_SIZE ) )

        if epoch in [2, 5]:
            learning_rate = learning_rate/100

        for i in tqdm(range(num_batches)):
            xb, yb = data.get_batch()
            loss_np, _, summary = sess.run([loss, optim, merged_summary_op],
                feed_dict={x: xb, y: yb, phase: True, lr: learning_rate})
            # logs every batch
            summary_writer.add_summary(summary, epoch * num_batches + i)
            avg_cost  += loss_np/num_batches

        # Display logs per epoch step
        if (epoch+1) % display_epoch == 0:
            print("Epoch:", '%02d' % (epoch+1),
                "cost=", "{:.6f}".format(avg_cost))

        print('Validation Set Accuracy:',
            accuracy.eval({x: data.x_val, y: data.y_val, phase: False}))

    # Test the model on separate data
    print('Test Set Accuracy:',
        accuracy.eval({x: data.x_test, y: data.y_test, phase: False}))

    print("Run the command line:\n--> tensorboard --logdir=./tf_logs ")
