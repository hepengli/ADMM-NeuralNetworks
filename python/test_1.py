from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.contrib.eager as tfe
from network import ADMM_NN
from tensorflow.examples.tutorials.mnist import input_data

# Parameters
n_inputs = 28*28 # MNIST image shape 28*28
n_outputs = 10  # MNIST classes from 0-9 digits
n_hiddens = [256, 256]  # number of neurons
n_batches = 5000
n_friends = 1

train_epochs = 1000

beta = 5.0
rho = 100.0
gamma = 5.0


# Load MNIST data
mnist = input_data.read_data_sets("./data/", one_hot=True)

trainX = mnist.train.images.astype(np.float32)[:n_batches,:]
trainY = mnist.train.labels.astype(np.float32)[:n_batches,:]

# Initial Model
model = ADMM_NN(n_inputs, n_hiddens, n_outputs, n_friends, n_batches)
mu, vf = model.feedforward(trainX)
train_op = model.fit_vf(trainY, beta, gamma)
loss_op = model.evaluate(trainX, trainY.transpose())

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Training Model
for i in range(train_epochs):
    print("------ Training: {:d} ------".format(i))
    sess.run(train_op)
    loss = sess.run(loss_op)
    print("Loss train: %3f" % (np.array(loss)))


    s = sess.run(model.s)
    w0 = sess.run(model.v[0])
    w1 = sess.run(model.v[1])
    w2 = sess.run(model.v[2])
    x0 = sess.run(model.vx[0])
    x1 = sess.run(model.vx[1])
    o0 = sess.run(model.vo[0])
    o1 = sess.run(model.vo[1])
    a = sess.run(model.vx[2])

    print('max_w2: ', np.abs(w2).max())
    print('max_o1: ', np.abs(o1).max())
    print('w0*s - x0: ', np.abs(w0.dot(s)-x0).max())
    print('w1*o0 - x1: ', np.abs(w1.dot(o0)-x1).max())
    print('w2*o1 - a: ', np.abs(w2.dot(o1)-a).max())
    input()