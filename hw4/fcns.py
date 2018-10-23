import os
import pickle
import numpy as np

num_classes = 10

# helper functions
def dim(dim, cp):
    if(cp['kpad'] == "SAME"):
        c = int( np.ceil(float(dim) / float(cp['ks']) ) )
    else:
        c = int( np.ceil(float(dim - cp['k'] + 1) / float(cp['ks']) ) )
    if(cp['ppad'] == "SAME"):
        return int( np.ceil(float(c) / float(cp['ps']) ) )
    else:
        return int( np.ceil(float(c - cp['p'] + 1) / float(cp['ps']) ) )


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

def mkImg_2channel(batch_row):
    imgs3 = [np.empty((32,32,2)) for _ in range(3)]
    channels = [[0, 1], [0, 2], [1, 2]]
    for i in range(3):
        for chan in range(2):
            for row in range(32):
                c = 1024*channels[i][chan]
                r = 32*row
                imgs3[i][row,:,chan] = batch_row[c+r:c+r+32] / 255.0
    return imgs3

def mkImgs_2channel(batch_data):
    N = batch_data.shape[0]
    imgs3 = np.empty((3*N,32,32,3))
    for row in range(N):
        imgs3 = mkImg_2channel(batch_data[row,:])
        for i in range(3):
            imgs[row+i,:,:,:] = imgs3[i]
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
