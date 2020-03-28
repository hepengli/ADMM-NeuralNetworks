from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class AS_ADMM_NN(object):
    """ Class for ADMM Neural Network. """

    def __init__(self, n_inputs, n_hiddens, n_outputs, n_agents, n_edges, n_batches):

        """
        Initialize variables for NN.
        Raises:
            ValueError: Column input samples, for example, the input size of MNIST data should be (28x28, *) instead of (*, 28x28).
        :param n_inputs: Number of inputs.
        :param n_hiddens: Number of hidden units.
        :param n_outputs: Number of outputs
        :param n_agents: Number of agents
        :param n_batches: Number of data sample for each agent to train
        :param return:
        """
        self.a0 = np.zeros((n_agents, n_inputs, n_batches), dtype=np.float32)

        self.x1 = np.zeros((n_edges, n_agents, n_hiddens, n_inputs), dtype=np.float32)
        self.x2 = np.zeros((n_edges, n_agents, n_hiddens, n_hiddens), dtype=np.float32)
        self.x3 = np.zeros((n_edges, n_agents, n_outputs, n_hiddens), dtype=np.float32)

        self.w1 = np.zeros((n_agents, n_hiddens, n_inputs), dtype=np.float32)
        self.w2 = np.zeros((n_agents, n_hiddens, n_hiddens), dtype=np.float32)
        self.w3 = np.zeros((n_agents, n_outputs, n_hiddens), dtype=np.float32)

        self.y1 = np.zeros((n_edges, n_agents, n_hiddens, n_inputs), dtype=np.float32)
        self.y2 = np.zeros((n_edges, n_agents, n_hiddens, n_hiddens), dtype=np.float32)
        self.y3 = np.zeros((n_edges, n_agents, n_outputs, n_hiddens), dtype=np.float32)

        self.z1 = np.random.rand(n_agents, n_hiddens, n_batches).astype(np.float32)
        self.a1 = np.random.rand(n_agents, n_hiddens, n_batches).astype(np.float32)

        self.z2 = np.random.rand(n_agents, n_hiddens, n_batches).astype(np.float32)
        self.a2 = np.random.rand(n_agents, n_hiddens, n_batches).astype(np.float32)

        self.z3 = np.random.rand(n_agents, n_outputs, n_batches).astype(np.float32)

        self.lambda_larange = np.ones((n_agents, n_outputs, n_batches), dtype=np.float32)


    def _relu(self, x):
        """
        Relu activation function
        :param x: input x
        :return: max 0 and x
        """
        return tf.maximum(0.,x)

    def _weight_update(self, layer_output, activation_input, beta, rho, y, x, A):
        """
        Consider it now the minimization of the problem with respect to W_l.
        For each layer l, the optimal solution minimizes ||z_l - W_l a_l-1||^2. This is simply
        a least square problem, and the solution is given by W_l = z_l p_l-1, where p_l-1
        represents the pseudo-inverse of the rectangular activation matrix a_l-1.
        :param layer_output: output matrix (z_l)
        :param activation_input: activation matrix l-1  (a_l-1)
        :return: weight matrix
        """
        # p1, p2 = [], []
        # for n in range(self.a0.shape[0]):
        #     p1.append(tf.matmul(layer_output[n], activation_input[n], transpose_b=True))
        #     p2.append(tf.matmul(activation_input[n], activation_input[n], transpose_b=True))

        # p1, p2 = sum(p1), sum(p2)
        # p2 = p2 + np.eye(p2.shape[0])*1e-5
        # weight_matrix = tf.matmul(tf.cast(p1, tf.float32), tf.cast(np.linalg.inv(p2), tf.float32))

        m1 = beta * tf.matmul(layer_output, activation_input, transpose_b=True) - A * y + A * rho * x
        m2 = beta * tf.matmul(activation_input, activation_input, transpose_b=True) + rho * np.eye(activation_input.shape[0])
        weight_matrix = tf.matmul(tf.cast(m1, tf.float32), tf.cast(np.linalg.inv(m2), tf.float32))

        return weight_matrix

    def _activation_update(self, next_weight, next_layer_output, layer_nl_output, beta, gamma):
        """
        Minimization for a_l is a simple least squares problem similar to the weight update.
        However, in this case the matrix appears in two penalty terms in the problem, and so
        we must minimize:
            beta ||z_l+1 - W_l+1 a_l||^2 + gamma ||a_l - h(z_l)||^2
        :param next_weight:  weight matrix l+1 (w_l+1)
        :param next_layer_output: output matrix l+1 (z_l+1)
        :param layer_nl_output: activate output matrix h(z) (h(z_l))
        :param beta: value of beta
        :param gamma: value of gamma
        :return: activation matrix
        """
        # Calculate ReLU
        layer_nl_output = self._relu(layer_nl_output)

        # Activation inverse
        m1 = beta*tf.matmul(tf.matrix_transpose(next_weight), next_weight)
        m2 = tf.scalar_mul(gamma, tf.eye(tf.cast(m1.get_shape()[0], tf.int32)))
        av = tf.matrix_inverse(tf.cast(m1, tf.float32) + tf.cast(m2, tf.float32))

        # Activation formulate
        m3 = beta*tf.matmul(tf.matrix_transpose(next_weight), next_layer_output)
        m4 = gamma * layer_nl_output
        af = tf.cast(m3, tf.float32) + tf.cast(m4, tf.float32)

        # Output
        return tf.matmul(av, af)

    def _argminz(self, a, w, a_in, beta, gamma):
        """
        This problem is non-convex and non-quadratic (because of the non-linear term h).
        Fortunately, because the non-linearity h works entry-wise on its argument, the entries
        in z_l are decoupled. This is particularly easy when h is piecewise linear, as it can
        be solved in closed form; common piecewise linear choices for h include rectified
        linear units (ReLUs), that its used here, and non-differentiable sigmoid functions.
        :param a: activation matrix (a_l)
        :param w:  weight matrix (w_l)
        :param a_in: activation matrix l-1 (a_l-1)
        :param beta: value of beta
        :param gamma: value of gamma
        :return: output matrix
        """
        m = tf.matmul(tf.cast(w, tf.float32), tf.cast(a_in, tf.float32))
        sol1 = (gamma*a + beta*m)/(gamma + beta)
        sol2 = m
        z1 = np.zeros_like(a)
        z2 = np.zeros_like(a)
        z  = np.zeros_like(a)

        sol1 = np.array(sol1)
        sol2 = np.array(sol2)

        z1[sol1>=0.] = sol1[sol1>=0.]
        z2[sol2<=0.] = sol2[sol2<=0.]

        fz_1 = tf.square(gamma * (a - self._relu(z1))) + beta * (tf.square(z1 - m))
        fz_2 = tf.square(gamma * (a - self._relu(z2))) + beta * (tf.square(z2 - m))

        fz_1 = np.array(fz_1)
        fz_2 = np.array(fz_2)

        index_z1 = fz_1<=fz_2
        index_z2 = fz_2<fz_1

        z[index_z1] = z1[index_z1]
        z[index_z2] = z2[index_z2]

        return z

    def _argminlastz(self, targets, eps, w, a_in, beta):
        """
        Minimization of the last output matrix, using the above function.
        :param targets: target matrix (equal dimensions of z) (y)
        :param eps: lagrange multiplier matrix (equal dimensions of z) (lambda)
        :param w: weight matrix (w_l)
        :param a_in: activation matrix l-1 (a_l-1)
        :param beta: value of beta
        :return: output matrix last layer
        """
        m = tf.matmul(tf.cast(w, tf.float32), tf.cast(a_in, tf.float32))
        z = (targets - eps + beta*m)/(1+beta)
        return z

    def _lambda_update(self, zl, w, a_in, beta):
        """
        Lagrange multiplier update.
        :param zl: output matrix last layer (z_L)
        :param w: weight matrix last layer (w_L)
        :param a_in: activation matrix l-1 (a_L-1)
        :param beta: value of beta
        :return: lagrange update
        """
        mpt = tf.matmul(tf.cast(w, tf.float32), tf.cast(a_in, tf.float32))
        lambda_up = beta*(zl-mpt)
        return lambda_up

    def feed_forward(self, inputs):
        """
        Calculate feed forward pass for neural network
        :param inputs: inputs features
        :return: value of prediction
        """
        n = np.random.choice(self.a0.shape[0])
        outputs = self._relu(tf.matmul(self.w1[n], inputs))
        outputs = self._relu(tf.matmul(self.w2[n], outputs))
        outputs = tf.matmul(self.w3[n], outputs)
        return outputs

    def fit(self, inputs, labels, beta, gamma, rho, A):
        """
        Training ADMM Neural Network by minimizing sub-problems
        :param inputs: input of training data samples
        :param outputs: label of training data samples
        :param epochs: number of epochs
        :param beta: value of beta
        :param gamma: value of gamma
        :return: loss value
        """
        self.a0 = inputs.copy()

        # for m in range(self.x1.shape[0]):
        m = np.random.choice(A.shape[0])
        q = np.where(A[m] != 0)[0]
        # a)
        for n in q:
            # Input layer
            self.w1[n] = self._weight_update(self.z1[n], self.a0[n], beta, rho, self.y1[m,n], self.x1[m,n], A[m,n])
            self.a1[n] = self._activation_update(self.w2[n], self.z2[n], self.z1[n], beta, gamma)
            self.z1[n] = self._argminz(self.a1[n], self.w1[n], self.a0[n], beta, gamma)
            # Hidden layer
            self.w2[n] = self._weight_update(self.z2[n], self.a1[n], beta, rho, self.y2[m,n], self.x2[m,n], A[m,n])
            self.a2[n] = self._activation_update(self.w3[n], self.z3[n], self.z2[n], beta, gamma)
            self.z2[n] = self._argminz(self.a2[n], self.w2[n], self.a1[n], beta, gamma)
            # Output layer
            self.w3[n] = self._weight_update(self.z3[n], self.a2[n], beta, rho, self.y3[m,n], self.x3[m,n], A[m,n])
            self.z3[n] = self._argminlastz(labels[n], self.lambda_larange[n], self.w3[n], self.a2[n], beta)
            self.lambda_larange[n] = self._lambda_update(self.z3[n], self.w3[n], self.a2[n], beta)
        # b)
        v1 = 0.5 * (self.y1[m,q[0]] + self.y1[m,q[1]]) + \
            0.5 * rho * (A[m,q[0]]*self.w1[q[0]] + A[m,q[1]]*self.w1[q[1]])
        v2 = 0.5 * (self.y2[m,q[0]] + self.y2[m,q[1]]) + \
            0.5 * rho * (A[m,q[0]]*self.w2[q[0]] + A[m,q[1]]*self.w2[q[1]])
        v3 = 0.5 * (self.y3[m,q[0]] + self.y3[m,q[1]]) + \
            0.5 * rho * (A[m,q[0]]*self.w3[q[0]] + A[m,q[1]]*self.w3[q[1]])
        for n in q:
            self.x1[m,n] = (1.0/rho) * (self.y1[m,n] - v1) + A[m,n] * self.w1[n]
            self.x2[m,n] = (1.0/rho) * (self.y2[m,n] - v2) + A[m,n] * self.w2[n]
            self.x3[m,n] = (1.0/rho) * (self.y3[m,n] - v3) + A[m,n] * self.w3[n]
        # c)
            self.y1[m,n] = v1
            self.y2[m,n] = v2
            self.y3[m,n] = v3

        inputs = np.vstack(np.transpose(inputs, axes=[0,2,1])).transpose()
        labels = np.vstack(np.transpose(labels, axes=[0,2,1])).transpose()
        loss, accuracy = self.evaluate(inputs, labels)
        return loss, accuracy

    def evaluate(self, inputs, labels, isCategrories = True ):
        """
        Calculate loss and accuracy (only classification)
        :param inputs: inputs data
        :param outputs: ground truth
        :param isCategrories: classification or not
        :return: loss and accuracy (only classification)
        """
        forward = self.feed_forward(inputs)
        loss = tf.reduce_mean(tf.square(forward - labels))

        if isCategrories:
            accuracy = tf.equal(tf.argmax(labels, axis=0), tf.argmax(forward, axis=0))
            accuracy = tf.reduce_sum(tf.cast(accuracy, tf.int32)) / accuracy.get_shape()[0]

        else:
            accuracy = loss

        return loss, accuracy

    def warming(self, inputs, labels, epochs, beta, gamma, rho, A):
        """
        Warming ADMM Neural Network by minimizing sub-problems without update lambda
        :param inputs: input of training data samples
        :param outputs: label of training data samples
        :param epochs: number of epochs
        :param beta: value of beta
        :param gamma: value of gamma
        :return:
        """
        self.a0 = inputs
        for i in range(epochs):
            print("------ Warming: {:d} ------".format(i))

            for m in range(self.x1.shape[0]):
            # m = np.random.choice(A.shape[0])
                q = np.where(A[m] != 0)[0]
                # a)
                for n in q:
                    # Input layer
                    self.w1[n] = self._weight_update(self.z1[n], self.a0[n], beta, rho, self.y1[m,n], self.x1[m,n], A[m,n])
                    self.a1[n] = self._activation_update(self.w2[n], self.z2[n], self.z1[n], beta, gamma)
                    self.z1[n] = self._argminz(self.a1[n], self.w1[n], self.a0[n], beta, gamma)
                    # Hidden layer
                    self.w2[n] = self._weight_update(self.z2[n], self.a1[n], beta, rho, self.y2[m,n], self.x2[m,n], A[m,n])
                    self.a2[n] = self._activation_update(self.w3[n], self.z3[n], self.z2[n], beta, gamma)
                    self.z2[n] = self._argminz(self.a2[n], self.w2[n], self.a1[n], beta, gamma)
                    # Output layer
                    self.w3[n] = self._weight_update(self.z3[n], self.a2[n], beta, rho, self.y3[m,n], self.x3[m,n], A[m,n])
                    self.z3[n] = self._argminlastz(labels[n], self.lambda_larange[n], self.w3[n], self.a2[n], beta)
                    self.lambda_larange[n] = self._lambda_update(self.z3[n], self.w3[n], self.a2[n], beta)
                # b)
                v1 = 0.5 * (self.y1[m,q[0]] + self.y1[m,q[1]]) + \
                    0.5 * rho * (A[m,q[0]]*self.w1[q[0]] + A[m,q[1]]*self.w1[q[1]])
                v2 = 0.5 * (self.y2[m,q[0]] + self.y2[m,q[1]]) + \
                    0.5 * rho * (A[m,q[0]]*self.w2[q[0]] + A[m,q[1]]*self.w2[q[1]])
                v3 = 0.5 * (self.y3[m,q[0]] + self.y3[m,q[1]]) + \
                    0.5 * rho * (A[m,q[0]]*self.w3[q[0]] + A[m,q[1]]*self.w3[q[1]])
                for n in q:
                    self.x1[m,n] = (1.0/rho) * (self.y1[m,n] - v1) + A[m,n] * self.w1[n]
                    self.x2[m,n] = (1.0/rho) * (self.y2[m,n] - v2) + A[m,n] * self.w2[n]
                    self.x3[m,n] = (1.0/rho) * (self.y3[m,n] - v3) + A[m,n] * self.w3[n]
                # c)
                    self.y1[m,n] = v1
                    self.y2[m,n] = v2
                    self.y3[m,n] = v3


    def drawcurve(self, train_, valid_, id, legend_1, legend_2):
        acc_train = np.array(train_).flatten()
        acc_test = np.array(valid_).flatten()

        plt.figure(id)
        plt.plot(acc_train)
        plt.plot(acc_test)

        plt.legend([legend_1, legend_2], loc='upper left')
        plt.draw()
        plt.pause(0.001)
        return 0
