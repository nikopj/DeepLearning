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
BATCH_SIZE = 200
NUM_EPOCHS = 15
learning_rate = .01
display_epoch = 1
# regularization parameters
drop_prob = 0.1
reg_scale = 1e-6

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
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=reg_scale)
        )
        x = tf.layers.batch_normalization(x, training=is_training, renorm=False)
        x = tf.nn.relu6(x)
        x = tf.layers.average_pooling2d(x, pool_size, p_strides, padding='same')
        x = tf.layers.dropout(x, rate=drop_prob, training=is_training)
        return x

    # --- FULLY CONNECTED LAYER WRAPPER ---
    # matmul -> dropout -> BN -> relu
    def fc_layer(input, units, is_training=True):
        x = tf.layers.dense(
            input, units,
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=reg_scale)
        )
        x = tf.layers.batch_normalization(x, training=is_training, renorm=False)
        x = tf.nn.relu6(x)
        x = tf.layers.dropout(x, rate=drop_prob, training=is_training)
        return x

    def B33(input, k_factor, is_training=True):
        in_shape = input.get_shape().as_list()
        x = tf.layers.batch_normalization(input, training=is_training, renorm=False)
        x = tf.nn.relu6(x)
        x = conv2d_fixed_padding(x, 16, 3, 1)
        # WIDE BOIS
        x = conv2d_fixed_padding(x, 16*k_factor, 3, 1)
        x = conv2d_fixed_padding(x, 32*k_factor, 3, 2)
        x = conv2d_fixed_padding(x, 64*k_factor, 3, 2)
        x = tf.layers.average_pooling2d(x, 8, 1, padding='valid')

        out_shape = x.get_shape().as_list()
        strides = in_shape[1] // out_shape[1]
        shortcut = conv2d_fixed_padding(input, out_shape[-1], 1, strides)
        return tf.add(shortcut, x)

    x = tf.placeholder(tf.float32, [None,input_size,input_size,num_channels])
    y = tf.placeholder(tf.float32, [None,num_classes])
    phase = tf.placeholder(tf.bool) # is_training

    # ---- NN -----

    def f(x):
        x = B33(x,1, is_training=True)
        print(x.get_shape())
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

def fixed_padding(inputs, kernel_size, data_format='channels_last'):
  """Pads the input along the spatial dimensions independently of input size.
  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
                 Should be a positive integer.
    data_format: The input format ('channels_last' or 'channels_first').
  Returns:
    A tensor with the same format as the input with the data either intact
    (if kernel_size == 1) or padded (if kernel_size > 1).
  """
  pad_total = kernel_size - 1
  pad_beg = pad_total // 2
  pad_end = pad_total - pad_beg

  if data_format == 'channels_first':
    padded_inputs = tf.pad(inputs, [[0, 0], [0, 0],
                                    [pad_beg, pad_end], [pad_beg, pad_end]])
  else:
    padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                    [pad_beg, pad_end], [0, 0]])
  return padded_inputs


def conv2d_fixed_padding(inputs, filters, kernel_size, strides, data_format='channels_last'):
  """Strided 2-D convolution with explicit padding."""
  # The padding is consistent and is based only on `kernel_size`, not on the
  # dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).
  if strides > 1:
    inputs = fixed_padding(inputs, kernel_size, data_format=data_format)

  return tf.layers.conv2d(
      inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
      padding=('SAME' if strides == 1 else 'VALID'), use_bias=False,
      kernel_initializer=tf.variance_scaling_initializer(),
      data_format=data_format)

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
