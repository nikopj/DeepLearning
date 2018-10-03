from fcns import *
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt

logs_path = "./tf_logs/"

# hyper-parameters
input_size = 32 # 32x32 imgs
num_channels = 3 # RGB
num_classes = 10
dim_layer = [num_channels, 32, 32, 32, 32, 32, 32, 64, 64, num_classes]
cp = { # conv parameters
    1:{
        'k':5, # conv window size (kxk)
        'p':2, # pool window size (pxp)
        'ks':2, # conv stride
        'ps':1, # pool stride,
        'kpad':"SAME",
        'ppad':"SAME"
    },
    2:{
        'k':3, # conv window size (kxk)
        'p':2, # pool window size (pxp)
        'ks':1, # conv stride
        'ps':2, # pool stride,
        'kpad':"SAME",
        'ppad':"SAME"
    },
    3:{
        'k':3, # conv window size (kxk)
        'p':2, # pool window size (pxp)
        'ks':1, # conv stride
        'ps':1, # pool stride,
        'kpad':"VALID",
        'ppad':"VALID"
    },
    4:{
        'k':3, # conv window size (kxk)
        'p':2, # pool window size (pxp)
        'ks':1, # conv stride
        'ps':1, # pool stride,
        'kpad':"VALID",
        'ppad':"VALID"
    },
    5:{
        'k':3, # conv window size (kxk)
        'p':2, # pool window size (pxp)
        'ks':1, # conv stride
        'ps':1, # pool stride,
        'kpad':"SAME",
        'ppad':"SAME"
    },
    6:{
        'k':3, # conv window size (kxk)
        'p':2, # pool window size (pxp)
        'ks':1, # conv stride
        'ps':1, # pool stride,
        'kpad':"SAME",
        'ppad':"SAME"
    }
}

d1 = dim(input_size,cp[1])
d2 = dim(d1,cp[2])
d3 = dim(d2,cp[3])
d4 = dim(d3,cp[4])
d5 = dim(d4,cp[5])
d6 = dim(d4,cp[6])
print(d1,d2,d3,d4,d5,d6)

BATCH_SIZE = 300
NUM_EPOCHS = 30
learning_rate = .1
display_epoch = 1

# regularization parameters
l2_lambda = .01/(sum([dim**2 for dim in dim_layer]))
print(l2_lambda)
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
        layer_3 = tf.nn.dropout( conv_layer(layer_2, W[3], b[3], cp[3]), keep_prob )
        layer_4 = tf.nn.dropout( conv_layer(layer_3, W[4], b[4], cp[4]), keep_prob )
        layer_5 = tf.nn.dropout( conv_layer(layer_4, W[5], b[5], cp[5]), keep_prob )
        layer_6 = tf.nn.dropout( conv_layer(layer_5, W[6], b[6], cp[6]), keep_prob )
        layer_7 = tf.nn.dropout( fc_layer(layer_6, W[7], b[7]), keep_prob )
        layer_8 = tf.nn.dropout( fc_layer(layer_7, W[8], b[8]), keep_prob )
        return tf.squeeze(
            tf.add( tf.matmul(layer_8, W['out']), b['out'] )
        )

    # Store layers weight & bias
    # default dtype=float32
    # WEIGHTS
    W = {
        1: tf.Variable(tf.random_normal( [cp[1]['k'],cp[1]['k'], num_channels, dim_layer[1]] )),
        2: tf.Variable(tf.random_normal( [cp[2]['k'],cp[2]['k'], dim_layer[1], dim_layer[2]] )),
        3: tf.Variable(tf.random_normal( [cp[3]['k'],cp[3]['k'], dim_layer[2], dim_layer[3]] )),
        4: tf.Variable(tf.random_normal( [cp[4]['k'],cp[4]['k'], dim_layer[3], dim_layer[4]] )),
        5: tf.Variable(tf.random_normal( [cp[5]['k'],cp[5]['k'], dim_layer[4], dim_layer[5]] )),
        6: tf.Variable(tf.random_normal( [cp[6]['k'],cp[6]['k'], dim_layer[5], dim_layer[6]] )),
        7: tf.Variable(tf.random_normal( [d6*d6*dim_layer[6], dim_layer[7]] )),
        8: tf.Variable(tf.random_normal( [dim_layer[7], dim_layer[8]] )),
        'out': tf.Variable(tf.random_normal( [dim_layer[8], num_classes] ))
    }
    # BIASES
    b = {
        1: tf.Variable(tf.random_normal([dim_layer[1]])),
        2: tf.Variable(tf.random_normal([dim_layer[2]])),
        3: tf.Variable(tf.random_normal([dim_layer[3]])),
        4: tf.Variable(tf.random_normal([dim_layer[4]])),
        5: tf.Variable(tf.random_normal([dim_layer[5]])),
        6: tf.Variable(tf.random_normal([dim_layer[6]])),
        7: tf.Variable(tf.random_normal([dim_layer[7]])),
        8: tf.Variable(tf.random_normal([dim_layer[8]])),
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
def conv_layer(x, W, b, cp):
    x = tf.nn.conv2d(x, W, strides=[1, cp['ks'], cp['ks'], 1],
        padding=cp['kpad'])
    x = tf.add(x,b)
    mean, var = tf.nn.moments(x, axes=[0,1,2])
    x = tf.nn.batch_normalization(x, mean, var, 0, 1, .001)
    x = tf.nn.relu6(x)
    return tf.nn.avg_pool(x, ksize = [1, cp['p'], cp['p'], 1],
        strides = [1, cp['ps'], cp['ps'], 1], padding=cp['ppad'])

# fully connected layer
def fc_layer(x, W, b):
    x = tf.add( tf.matmul( tf.reshape(x, [-1, tf.shape(W)[0]]), W ), b)
    mean, var = tf.nn.moments(x, axes=[0])
    x = tf.nn.batch_normalization(x, mean, var, 0, 1, .001)
    return tf.nn.relu6(x)

if __name__ == '__main__':
    main()
