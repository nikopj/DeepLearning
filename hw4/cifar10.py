import os
import pickle
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt

logs_path = "./tf_logs/"

# hyper-parameters
input_size = 32 # 32x32 imgs
num_channels = 3 # RGB
num_classes = 10
dim_layer = {1:32, 2:64, 3:64, 4:num_classes}
cp = { # conv parameters
    1:{
        'k':5, # conv window size (kxk)
        'p':2, # pool window size (pxp)
        'ks':2, # conv stride
        'ps':2 # pool stride
    },
    2:{
        'k':5, # conv window size (kxk)
        'p':2, # pool window size (pxp)
        'ks':2, # conv stride
        'ps':2 # pool stride
    }
}
def dim(dim, cp):
    c = int( np.ceil(float(dim - cp['k'] + 1) / float(cp['ks']) ) )
    return int( np.ceil(float(c - cp['p'] + 1) / float(cp['ps']) ) )

d1 = dim(input_size,cp[1])
d2 = dim(d1,cp[2])

# NUM_BATCHES = 300
BATCH_SIZE = 300
NUM_EPOCHS = 8 
learning_rate = 0.1
display_epoch = 1

# regularization parameters
l2_lambda = .01/( d1*d1*dim_layer[1] + d2*d2*dim_layer[2] + dim_layer[2]*dim_layer[3] )
drop_out = .9

def main():
    class Data(object):
        def __init__(self):
            np.random.seed(31415)
            ([self.x_train, self.x_val, self.x_test],
                [self.y_train, self.y_val, self.y_test])  = \
                loadData("./cifar10_data")

            self.index = np.arange(self.x_train.shape[0])

        def get_batch(self):
            choices = np.random.choice(self.index, size=BATCH_SIZE)
            return self.x_train[choices,:,:,:], self.y_train[choices,:]

    x = tf.placeholder(tf.float32, [None,input_size,input_size,num_channels])
    y = tf.placeholder(tf.float32, [None,num_classes])
    keep_prob = tf.placeholder(tf.float32) # for dropout

    # f:R28x28 -> R10
    def f(x):
        layer_1 = tf.nn.dropout( conv_layer(x, W[1], b[1], cp[1]), keep_prob )
        layer_2 = tf.nn.dropout( conv_layer(layer_1, W[2], b[2], cp[2]), keep_prob )
        layer_3 = tf.nn.dropout( fc_layer(layer_2, W[3], b[3]), keep_prob )
        return tf.squeeze(
            tf.add( tf.matmul(layer_3, W['out']), b['out'] )
        )

    # Store layers weight & bias
    # default dtype=float32
    # WEIGHTS
    W = {
        1: tf.Variable(tf.random_normal( [cp[1]['k'],cp[1]['k'], num_channels, dim_layer[1]] )),
        2: tf.Variable(tf.random_normal( [cp[2]['k'],cp[2]['k'], dim_layer[1], dim_layer[2]] )),
        3: tf.Variable(tf.random_normal( [d2*d2*dim_layer[2], dim_layer[3]] )),
        'out': tf.Variable(tf.random_normal( [dim_layer[3], num_classes] ))
    }
    # BIASES
    b = {
        1: tf.Variable(tf.random_normal([dim_layer[1]])),
        2: tf.Variable(tf.random_normal([dim_layer[2]])),
        3: tf.Variable(tf.random_normal([dim_layer[3]])),
        'out': tf.Variable(tf.random_normal([num_classes]))
    }

    # models
    logits = f(x)
    prediction = tf.nn.softmax(logits)
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # cross entropy loss with L2 penalty on weights
    loss = tf.reduce_mean( tf.losses.softmax_cross_entropy(y, logits) ) + \
        l2_lambda*tf.reduce_sum(
            [tf.nn.l2_loss(var) for var in
            tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)]
        )
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
            num_batches = ceil(data.index[-1]/BATCH_SIZE)
            for i in tqdm(range(num_batches)):
                xb, yb = data.get_batch()
                loss_np, _, summary = sess.run([loss, optim, merged_summary_op],
                    feed_dict={x: xb, y: yb, keep_prob: drop_out})
                # logs every batch
                summary_writer.add_summary(summary, epoch * num_batches + i)
                avg_cost  += loss_np/num_batches
            # Display logs per epoch step
            if (epoch+1) % display_epoch == 0:
                print("Epoch:", '%02d' % (epoch+1),
                    "cost=", "{:.6f}".format(avg_cost))
            print('Validation Set Accuracy:',
                accuracy.eval({x: data.x_val, y: data.y_val, keep_prob: 1.0}))

        # Test the model on separate data
        print('Test Set Accuracy:',
            accuracy.eval({x: data.x_test, y: data.y_test, keep_prob: 1.0}))

        print("Run the command line:\n--> tensorboard --logdir=./tf_logs ")

# -------------- MODEL FUNCTIONS -----------------

# convolution layer: conv-> +bias -> activation -> pool
def conv_layer(x, W, b, cp, phase=True):
    x = tf.nn.conv2d(x, W, strides=[1, cp['ks'], cp['ks'], 1],
        padding="VALID")
    x = tf.add(x,b)
    mean, var = tf.nn.moments(x, axes=[0,1,2])
    x = tf.nn.batch_normalization(x, mean, var, 0, 1, .001)
    x = tf.nn.relu6(x)
    return tf.nn.avg_pool(x, ksize = [1, cp['p'], cp['p'], 1],
        strides = [1, cp['ps'], cp['ps'], 1], padding="VALID")

# fully connected layer
def fc_layer(x, W, b):
    x = tf.add( tf.matmul( tf.reshape(x, [-1, tf.shape(W)[0]]), W ), b)
    mean, var = tf.nn.moments(x, axes=[0])
    x = tf.nn.batch_normalization(x, mean, var, 0, 1, .001)
    return tf.nn.relu6(x)

# -------------- DATA FUNCTIONS -----------------

# from https://www.cs.toronto.edu/~kriz/cifar.html
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
# from  https://stackoverflow.com/questions/38592324/one-hot-encoding-using-numpy/38592416
def get_one_hot(targets, nb_classes):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape)+[nb_classes])

def mkImg(batch_row):
    img = np.empty((32,32,3))
    for chan in range(3):
        for row in range(32):
            c = 1024*chan
            r = 32*row
            img[row,:,chan] = batch_row[c+r:c+r+32]
    return img / 255.0

def mkImgs(batch_data):
    N = batch_data.shape[0]
    imgs = np.empty((N,32,32,3))
    for row in range(N):
        imgs[row,:,:,:] = mkImg(batch_data[row,:])
    return imgs

def loadData(data_folder):
    x_train = np.empty((0,3072))
    y_train = np.empty((0,10))
    for fname in os.listdir(data_folder):
        if fname.find("data") >= 0 :
            train_batch = unpickle(data_folder+"/"+fname)
            x_train = np.vstack( (x_train, train_batch[b'data']) )
            y_train = np.vstack( (y_train,
                get_one_hot(np.array(train_batch[b'labels']), num_classes))
            )
        if fname.find("test") >= 0:
            test_batch = unpickle(data_folder+"/"+fname)
            [x_val, x_test] = np.split(mkImgs(test_batch[b'data']), 2)
            [y_val, y_test] = np.split(
                get_one_hot(np.array(test_batch[b'labels']), num_classes), 2
            )
    x_train = mkImgs(x_train)
    return ([x_train, x_val, x_test], [y_train, y_val, y_test])

if __name__ == '__main__':
    main()
