#!/bin/python3.6

import numpy as np
import tensorflow as tf

sess = tf.Session()

def array_builder_example(in_shape):
    a = []
    x = tf.random_normal(in_shape)

    for _ in range(10):
        a.append(sess.run(x))

    print(np.array(a).shape)

def bad_array_builder_example(in_shape):
    """
    - hard to do read
    - doesn't work with multidim inputs
    - numpy arrays use contiguous memory,
      so every vstack requires a malloc
    """
    a = np.array([])
    x = tf.random_normal(in_shape)

    for _ in range(10):
        if a.size>0:
            a = np.vstack([a, sess.run(x)])
        else:
            a = sess.run(x)

    print(a.shape)

print("good")
array_builder_example([8,5])
array_builder_example([8])
array_builder_example([])

print("bad")
bad_array_builder_example([8,5])
bad_array_builder_example([8])
bad_array_builder_example([])
