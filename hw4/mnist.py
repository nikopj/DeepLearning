# Nikola Janjusevic
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import time
from tqdm import tqdm
from numpy.random import *

logs_path = "./tf_logs/"

def dim(dim, cp):
    c = int( np.ceil(float(dim - cp['k'] + 1) / float(cp['ks']) ) )
    return int( np.ceil(float(c - cp['p'] + 1) / float(cp['ps']) ) )

# hyper-parameters
input_dim = 28
num_classes = 10
NUM_BATCHES = 200
BATCH_SIZE = 500
NUM_EPOCHS = 8 # not technically using epochs but im keeping it.
learning_rate = 0.01
display_epoch = 1

# regularization parameters
reg_scale = 0.01
dropout = .1

# https://stackoverflow.com/questions/38592324/one-hot-encoding-using-numpy/38592416
def get_one_hot(targets, nb_classes):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape)+[nb_classes])

class Data(object):
    def __init__(self):
        mnist = tf.keras.datasets.mnist
        # 60k training, 10k testing
        (x_train, y_train),(x_test, y_test) \
            = mnist.load_data()
        # normalize [0, 255] to [0,1]
        self.x_train = x_train.reshape(x_train.shape[0],28,28,1) / 255.0
        x_test  = x_test.reshape(x_test.shape[0],28,28,1) / 255.0
        [self.x_val, self.x_test] = np.split(x_test,2)
        # digits [0,9] to length 10 one_hot vectors
        self.y_train = get_one_hot(y_train, num_classes)
        y_test = get_one_hot(y_test, num_classes)
        [self.y_val, self.y_test] = np.split(y_test,2)

        self.index_train = np.arange(x_train.shape[0])
        self.index_test = np.arange(x_test.shape[0])

    def get_batch(self):
        choices = choice(self.index_train, size=BATCH_SIZE)
        return self.x_train[choices,:,:,:], self.y_train[choices,:]

# --- CONV LAYER WRAPPER --- w/ L2 regularization
# conv -> dropout -> BN -> relu -> max_pool
def conv_layer(input, filters, kernel_size, pool_size=2, c_strides=1, \
    p_strides=1, is_training=False):
    x = tf.layers.conv2d(
        input, filters, kernel_size, strides=c_strides,
        kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=reg_scale)
    )
    x = tf.layers.batch_normalization(x, training=is_training, renorm=True)
    x = tf.nn.relu6(x)
    x = tf.layers.average_pooling2d(x, pool_size, p_strides)
    x = tf.layers.dropout(x, rate=dropout, training=is_training)
    return x

# --- FULLY CONNECTED LAYER WRAPPER ---
# matmul -> dropout -> BN -> relu
def fc_layer(input, units, is_training=False):
    x = tf.layers.dense(
        input, units,
        kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=reg_scale),
    )
    x = tf.layers.batch_normalization(x, training=is_training, renorm=True)
    x = tf.nn.relu6(x)
    x = tf.layers.dropout(x, rate=dropout, training=is_training)
    return x

x = tf.placeholder(tf.float32, [None,28,28,1])
y = tf.placeholder(tf.float32, [None,10])
phase = tf.placeholder(tf.bool) # is_training

def f(x):
    x = conv_layer(x, 4, 5, pool_size=2, c_strides=2, p_strides=2,
        is_training=phase
    )
    x = tf.layers.flatten(x)
    x = fc_layer(x, 16, is_training=phase)
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
        + tf.losses.get_regularization_loss()

    optim = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
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
    # training
    for epoch in range(NUM_EPOCHS):
        avg_cost = 0.
        num_batches = NUM_BATCHES
        for i in tqdm(range(num_batches)):
            xb, yb = data.get_batch()
            loss_np, _, summary = sess.run([loss, optim, merged_summary_op],
                feed_dict={x: xb, y: yb, phase: True})
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
