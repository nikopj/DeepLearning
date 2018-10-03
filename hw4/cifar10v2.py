import os
import pickle
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt

logs_path = "./tf_logs/"

input_size = 32 # 32x32 imgs
num_channels = 3 # RGB
num_classes = 10

# hyper-parameters
BATCH_SIZE = 300
NUM_EPOCHS = 15
learning_rate = .01
display_epoch = 1
# regularization parameters
drop_prob = 0.0
l2_lambda = 1e-8

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


    # -------------- MODEL FUNCTIONS -----------------

    # --- CONV LAYER WRAPPER --- w/ L2 regularization
    # conv -> dropout -> BN -> relu -> max_pool
    def conv_layer(input, filters, kernel_size, pool_size=1, c_strides=1, \
        p_strides=1, is_training=True):
        x = tf.layers.conv2d(
            input, filters, kernel_size, strides=c_strides, padding='same',
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale= \
                .001/( (kernel_size**2)*filters )
            )
        )
        x = tf.layers.dropout(x, rate=drop_prob, training=is_training)
        x = tf.layers.batch_normalization(x, training=is_training, renorm=True)
        x = tf.nn.relu6(x)
        x = tf.layers.max_pooling2d(x, pool_size, p_strides, padding='same')
        return x

    # --- FULLY CONNECTED LAYER WRAPPER ---
    # matmul -> dropout -> BN -> relu
    def fc_layer(input, units, is_training=True):
        x = tf.layers.dense(
            input, units,
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale= \
                .01/( units**2 )
            ),
        )
        x = tf.layers.dropout(x, rate=drop_prob, training=is_training)
        x = tf.layers.batch_normalization(x, training=is_training, renorm=True)
        x = tf.nn.relu6(x)
        return x


    x = tf.placeholder(tf.float32, [None,input_size,input_size,num_channels])
    y = tf.placeholder(tf.float32, [None,num_classes])
    phase = tf.placeholder(tf.bool) # is_training

    # ---- NN -----

    def f(x):
        x = conv_layer(x, 16, 5, pool_size=2, c_strides=2, p_strides=2,
            is_training=phase
        )
        print(x.get_shape())
        x = conv_layer(x, 16, 3, pool_size=2, c_strides=1, p_strides=1,
            is_training=phase
        )
        print(x.get_shape())
        x = conv_layer(x, 16, 3, pool_size=2, c_strides=1, p_strides=2,
            is_training=phase
        )
        print(x.get_shape())
        x = conv_layer(x, 16, 1, pool_size=1, c_strides=1, p_strides=1,
            is_training=phase
        )
        print(x.get_shape())
        x = conv_layer(x, 16, 1, pool_size=1, c_strides=1, p_strides=2,
            is_training=phase
        )
        print(x.get_shape())
        x = tf.layers.flatten(x)
        x = fc_layer(x, 16, is_training=phase)
        x = tf.layers.dense(x, num_classes)
        return x

    # models
    logits = f(x)
    prediction = tf.nn.softmax(logits)

    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    loss = tf.reduce_mean( tf.losses.softmax_cross_entropy(y, logits) ) \
        + l2_lambda*tf.reduce_sum(
            [tf.nn.l2_loss(var) for var in
            tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)]
        )

    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(extra_update_ops):
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
            num_batches = int( np.ceil( data.index[-1] / BATCH_SIZE ) )

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
