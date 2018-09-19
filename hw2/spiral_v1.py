import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from numpy.random import *
import time
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D

# intermediate dimensional number
dim_in = 2
dim_layer1 = 10
dim_layer2 = 5
BATCH_SIZE = 250
NUM_BATCHES = 1000

class Data(object):
    def __init__(self):
        seed(int(time.time()))
        nsamp = 300
        self.index = np.arange(nsamp)

        var = .1
        # noise
        v = np.sqrt(var/2)*randn(nsamp,4)

        # class 1
        t1 = uniform(np.pi/4, 4*np.pi, nsamp)
        x1 = np.array([ t1*np.sin(t1)+v[:,0], t1*np.cos(t1)+v[:,1] ]).T

        # class 2
        t2 = uniform(np.pi/4, 4*np.pi, nsamp)
        x2 = np.array([ -t2*np.sin(t2)+v[:,2], -t2*np.cos(t2)+v[:,3] ]).T

        self.x = np.concatenate((x1,x2))
        self.y = np.concatenate((np.zeros(nsamp), np.ones(nsamp)))

    def get_batch(self):
        choices = choice(self.index, size=BATCH_SIZE)
        return self.x[choices,:], self.y[choices].flatten()

# Store layers weight & bias
# default dtype=float32
weights = {
    'w1': tf.Variable(tf.random_normal([dim_in, dim_layer1])),
    'w2': tf.Variable(tf.random_normal([dim_layer1, dim_layer2])),
    'out': tf.Variable(tf.random_normal([dim_layer2, 1]))
}
biases = {
    'b1': tf.Variable(tf.zeros([dim_layer1])),
    'b2': tf.Variable(tf.zeros([dim_layer2])),
    'out': tf.Variable(tf.zeros([1]))
}

def f(x):
    layer_1 = tf.add( tf.matmul(x, weights['w1']), biases['b1'] )
    layer_2 = tf.nn.relu6( tf.add( tf.matmul(layer_1, weights['w2']), biases['b2'] ))
    return tf.squeeze( tf.add( tf.matmul(layer_2, weights['out']), biases['out'] ) )

# f: R2 -> R
# def f(x):
#     # W1: R2 -> RM
#     W1  = tf.get_variable('W1', [2, M], tf.float32,
#                         tf.random_normal_initializer())
#     # W2: RM -> RM
#     W2  = tf.get_variable('W2', [M, M], tf.float32,
#                         tf.random_normal_initializer())
#     # w: RM -> R
#     w  =  tf.get_variable('w', [M, 1], tf.float32,
#                         tf.random_normal_initializer())
#     b1 = tf.get_variable('b1', [1, M], tf.float32,
#                         tf.zeros_initializer())
#     b2 = tf.get_variable('b2', [1, M], tf.float32,
#                         tf.zeros_initializer())
#     b3 = tf.get_variable('b3', [], tf.float32, tf.zeros_initializer())
#
#     return tf.squeeze(
#         tf.matmul( tf.nn.relu6( tf.matmul(
#             tf.nn.relu6( tf.matmul( x, W1 ) + b1 ), W2
#         ) + b2 ) , w ) + b3
#     )

x = tf.placeholder(tf.float32, [BATCH_SIZE,2])
y = tf.placeholder(tf.float32, [BATCH_SIZE])
y_hat = f(x)
# loss = tf.reduce_mean( tf.pow(-y*tf.log(y_hat) - (1-y)*tf.log(1-y_hat),1) \
#    + .9*(tf.norm(W1) + tf.norm(W2)) )
loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits( logits=y_hat, labels=y ) +
    .5*( tf.nn.l2_loss(weights['w1'])**2 + tf.nn.l2_loss(weights['w2'])**2 +
    tf.nn.l2_loss(weights['out'])**2 )
)
optim = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)
init = tf.global_variables_initializer()

# sess = tf.Session()
# sess.run(init)
#
# data = Data()
#
# for _ in tqdm(range(0, NUM_BATCHES)):
#     x_np, y_np = data.get_batch()
#     loss_np, _ = sess.run([loss, optim], feed_dict={x: x_np, y: y_np})
#
# for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
#     print(np.array(sess.run(var)))

# print(W1)
# print(W2)
# print(w)
# print(b1)
# print(b2)
# print(b3)

# N = BATCH_SIZE
# z = np.linspace(-15,15,N, dtype=np.float32)
# X1,X2 = np.meshgrid(z,z, indexing="ij")
# Y_hat = np.zeros((N,N), dtype=np.float32)
#
# for i in range(N):
#     xx = np.array( [X1[i,:], X2[i,:]] ).reshape((BATCH_SIZE,2))
#     Y_hat[i,:] = sess.run(y_hat, feed_dict={x: xx})
#
#
# # fig = plt.figure()
# # ax = fig.add_subplot(111, projection='3d')
# # ax.scatter(X1,X2,Y_hat)
# # plt.show()
