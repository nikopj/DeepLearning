import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import time
from tqdm import tqdm
from numpy.random import *

# intermediate dimensional number
dim_layer1 = 16
dim_layer2 = 32
num_classes = 10
NUM_BATCHES = 200
BATCH_SIZE = 500
NUM_EPOCHS = 15
learning_rate = 0.01

k1 = 5
pool1 = 2
k_stride1 = 2
p_stride1 = 2

l2_lambda = .01/(k1*k1*dim_layer1 + dim_layer2)
drop_out = .9

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
        self.x_test  = x_test.reshape(x_test.shape[0],28,28,1) / 255.0
        # digits [0,9] to length 10 one_hot vectors
        self.y_train = get_one_hot(y_train, num_classes)
        self.y_test = get_one_hot(y_test, num_classes)

        self.index_train = np.arange(x_train.shape[0])
        self.index_test = np.arange(x_test.shape[0])

    def get_batch(self):
        choices = choice(self.index_train, size=BATCH_SIZE)
        return self.x_train[choices,:,:,:], self.y_train[choices,:]

# convolution layer: conv-> +bias -> activation -> pool
def conv_layer(input, W, b, k_stride=1, pool=2, p_stride=2):
    x = tf.nn.conv2d(input, W, strides=[1, k_stride, k_stride, 1], padding="VALID")
    x = tf.add(x,b)
    x = tf.nn.relu6(x)
    return tf.nn.avg_pool(x, ksize = [1, pool, pool, 1],
        strides = [1, p_stride, p_stride, 1], padding="VALID")

# fully connected layer
def fc_layer(input, W, b):
    x = tf.add( tf.matmul( tf.reshape(input, [-1, tf.shape(W)[0]]), W ), b)
    return tf.nn.relu6(x)

x = tf.placeholder(tf.float32, [None,28,28,1])
y = tf.placeholder(tf.float32, [None,10])
keep_prob = tf.placeholder(tf.float32) # for dropout

# f:R28x28 -> R10
def f(x):
    layer_1 = tf.nn.dropout( conv_layer(x, weights['w1'], biases['b1'],
        k_stride=k_stride1, pool=pool1, p_stride=p_stride1), keep_prob )
    layer_2 = tf.nn.dropout( fc_layer(layer_1, weights['w2'], biases['b2']), keep_prob)
    return tf.squeeze(
        tf.add( tf.matmul(layer_2, weights['out']), biases['out'] )
    )

# Store layers weight & bias
# default dtype=float32
d1 = int( np.ceil(float(28 - k1 + 1) / float(k_stride1) ) )
dp1 = int( np.ceil(float(d1 - pool1 + 1) / float(p_stride1) ) )
d2 = int( np.ceil(float(dp1 - k2 + 1) / float(k_stride2) ) )
dp2 = int( np.ceil(float(d2 - pool2 + 1) / float(p_stride2) ) )
print(d1,dp1,d2,dp2)
weights = {
    'w1': tf.Variable(tf.random_normal([k1,k1,1,dim_layer1])), # kernel
    'w2': tf.Variable(tf.random_normal([dp1*dp1*dim_layer1,dim_layer2])),
    'out': tf.Variable(tf.random_normal([dim_layer2, num_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([dim_layer1])),
    'b2': tf.Variable(tf.random_normal([dim_layer2])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}

# models
logits = f(x)
prediction = tf.nn.softmax(logits)

correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# binar cross entropy loss with L2 penalty on weights
loss = tf.reduce_mean( tf.losses.softmax_cross_entropy(y, logits) ) + \
    l2_lambda*tf.reduce_sum(
        [tf.nn.l2_loss(var) for var in
        tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)]
    )
optim = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    data = Data()
    sess.run(init)
    # training
    for epoch in range(NUM_EPOCHS):
        avg_cost = 0.
        num_batches = NUM_BATCHES
        for _ in tqdm(range(num_batches)):
            xb, yb = data.get_batch()
            loss_np, _ = sess.run([loss, optim],
                feed_dict={x: xb, y: yb, keep_prob: drop_out})
            avg_cost  += loss_np/num_batches
        # Display logs per epoch step
        if (epoch+1) % display_epoch == 0:
            print("Epoch:", '%04d' % (epoch+1),
                "cost=", "{:.9f}".format(avg_cost))
        print('Accuracy:',
            accuracy.eval({x: data.x_test, y: data.y_test, keep_prob: 1.0}))
