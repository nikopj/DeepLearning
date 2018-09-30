import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# https://stackoverflow.com/questions/38592324/one-hot-encoding-using-numpy/38592416
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
    imgs = np.empty( (N,32,32,3) )
    for row in range(N):
        imgs[row,:,:,:] = mkImg(batch_data[row,:])

x_train, x_val, x_test = [np.empty((0,32*32*3)) for _ in range(3)]
y_train, y_val, y_test = [np.empty((0,10)) for _ in range(3)]
data_folder = "cifar10_data"
for fname in os.listdir(data_folder):
    if fname.find("data"):
        train_batch = unpickle(cifar10_folder+"/"+fname)
        x_train = np.vstack( (x_train, train_batch[b'data']) )
        y_train = np.vstack(
            (y_train, get_one_hot(train_batch[b'labels'], num_classes))
        )
    if fname.find("test"):
        test_batch = unpickle(cifar10_folder+"/"+fname)
        


# class Data(object):
#     def __init__(self):
#
#
#     def get_batch(self):
#         choices = choice(self.index_train, size=BATCH_SIZE)
#         return self.x_train[choices,:,:,:], self.y_train[choices,:]
