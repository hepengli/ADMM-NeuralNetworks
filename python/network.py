from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class ADMM_NN(object):
    """ Class for ADMM Neural Network. """

    def __init__(self, n_inputs, n_hiddens, n_outputs, n_friends, n_batches=None, dtype='float32'):
        """
        Initialize variables for NN.
        Raises:
            ValueError: Column input samples, for example, the input size of MNIST data should be (28x28, *) instead of (*, 28x28).
        :param n_inputs: Number of inputs
        :param n_hiddens: Number of hidden units.
        :param n_outputs: Number of outputs
        :param n_friends: Number of friends
        :param return:
        """
        n_hiddens = [n_hiddens] if not isinstance(n_hiddens, list) else n_hiddens
        self.n_nodes = [n_inputs] + n_hiddens
        self._dtype = dtype
        self.n_friends = n_friends
        self.n_outputs = n_outputs

        n_nodes = self.n_nodes + [n_outputs]
        with tf.variable_scope('pi', reuse=tf.AUTO_REUSE):
            # NN weights:
            self.w = [tf.get_variable('w{}'.format(i), shape=[n_nodes[i+1], n_nodes[i]], dtype=dtype) for i, _ in enumerate(n_nodes[:-1])]
            # Neurons' inputs and outputs
            self.x = [tf.get_variable('x{}'.format(i), shape=[n_node, n_batches], dtype=dtype) for i, n_node in enumerate(n_hiddens)]
            self.o = [tf.get_variable('o{}'.format(i), shape=[n_node, n_batches], dtype=dtype) for i, n_node in enumerate(n_hiddens)]
            # NN' outputs
            a = tf.get_variable('a', shape=[n_outputs, n_batches], dtype=dtype)
            self.lam = tf.get_variable('lam', shape=[n_outputs, n_batches], dtype=dtype)
            self.x.append(a)
            # Estimate of neighbours' outpus
            self.z = tf.get_variable('z', shape=[n_friends, n_outputs, n_batches], dtype=dtype)
            self.p = tf.get_variable('p', shape=[n_friends, n_outputs, n_batches], dtype=dtype)

        n_nodes = self.n_nodes + [10]
        with tf.variable_scope('vf', reuse=tf.AUTO_REUSE):
            # Value NN weights
            self.v = [tf.get_variable('v{}'.format(i), shape=[n_nodes[i+1], n_nodes[i]], dtype=dtype) for i, _ in enumerate(n_nodes[:-1])]
            # Neurons' inputs and outputs
            self.vx = [tf.get_variable('vx{}'.format(i), shape=[n_node, n_batches], dtype=dtype) for i, n_node in enumerate(n_hiddens)]
            self.vo = [tf.get_variable('vo{}'.format(i), shape=[n_node, n_batches], dtype=dtype) for i, n_node in enumerate(n_hiddens)]
            # NN' outputs
            vf = tf.get_variable('vf', shape=[n_nodes[-1], n_batches], dtype=dtype)
            self.vlam = tf.get_variable('vlam', shape=[n_nodes[-1], n_batches], dtype=dtype)
            self.vx.append(vf)

    def pinv(self, a, rcond=1e-15):
        s, u, v = tf.svd(a)
        # Ignore singular values close to zero to prevent numerical overflow
        limit = rcond * tf.reduce_max(s)
        non_zero = tf.greater(s, limit)

        reciprocal = tf.where(non_zero, tf.reciprocal(s), tf.zeros_like(s))
        lhs = tf.matmul(v, tf.matrix_diag(reciprocal))
        return tf.matmul(lhs, u, transpose_b=True)

    def _relu(self, x):
        return tf.maximum(0.,x)

    def _weight_update(self, x, o, beta):
        """
        min_{W_l} beta * ||x_l - W_l * o_{l-1}||^2

        W_l = (beta * x_l * O_{l-1}^T) * inv(beta * o_{l-1} * O_{l-1}^T)

        OR

        W_l = x_l * pinv

        return: W_l
        """
        return tf.matmul(x, self.pinv(o))

    def _activation_update(self, x_next, W_next, x, beta, gamma):
        """
        min_{o_l} beta * ||x_{l+1} - W_{l+1} * o_l||^2 + gamma * ||o_l - h(x_l)||^2

        o_l = inv(W_{l+1}^T * W_{l+1} + gamma * I) * (beta * W_{l+1}^T * x_{l+1} + gamma * h(x_l))

        return: o_l
        """
        # Activation inverse
        m1 = tf.matmul(W_next, W_next, transpose_a=True)
        m2 = tf.scalar_mul(gamma, tf.eye(int(m1.shape[0]), dtype=self._dtype))
        av = tf.linalg.inv(m1+m2)

        # Activation formulate
        m3 = beta * tf.matmul(W_next, x_next, transpose_a=True)
        m4 = gamma * self._relu(x)
        af = m3 + m4

        # Output
        return tf.matmul(av, af)

    def _argmin_x(self, o, w, o_last, beta, gamma):
        """
        min_{x_l} gamma * ||o_l - h(x_l)||^2 + beta * ||x_l - W_l * O_{l-1}||^2
        
        x_l = (gamma * o_l + beta * W_l * O_{l-1}) / (gamma + beta),  if x_l > 0

            =  W_l * O_{l-1}, otherwise
        """
        m = tf.matmul(w, o_last)
        sol1 = (gamma * o + beta * m) / (gamma + beta)
        sol2 = m
        x1 = tf.zeros_like(o)
        x2 = tf.zeros_like(o)

        x1 = tf.where(tf.less_equal(x1, sol1), sol1, x1)
        x2 = tf.where(tf.less_equal(sol2, x2), sol2, x2)

        f_1 = gamma * tf.square(o - self._relu(x1)) + beta * (tf.square(x1 - m))
        f_2 = gamma * tf.square(o - self._relu(x2)) + beta * (tf.square(x2 - m))

        return tf.where(tf.less_equal(f_1, f_2), x1, x2)

    def _argmin_a(self, g, w, o, z, p, lam, beta, rho, A):
        """
        min_{a} sum( g .* (a - a_k)) + beta * ||a - w * o||^2 + 
                p .* (A * a - z) + rho * ||A * a - z||^2
        
        # A should be 1 or -1

        return: a
        """
        m = - g - lam + beta * tf.matmul(w, o) - p * A + rho * A * z
        v = m / (beta + rho * A * A)

        # No friends
        if self.n_friends == 1:
            m = - g - lam + beta * tf.matmul(w, o)
            v = m / beta

        return v

    def _argmin_vf(self, targets, lam, v, o, beta):
        m = tf.matmul(v, o)
        v = (targets - lam + beta * m) / (1 + beta)

        return v

    def _lam_update(self, a, w, o, beta):
        return beta*(a - tf.matmul(w, o))

    def _z_and_p_update(self, a, a_neighbor, p, p_neighbor, A, A_neighbor, rho):
        v = 0.5 * (p + p_neighbor) + 0.5 * rho * (A * a + A_neighbor * a_neighbor)
        z = (1.0/rho) * (p - v) + A * a
        p = v
        return z, p

    def feedforward(self, normed_x):
        mu = tf.transpose(normed_x)
        for i, w in enumerate(self.w):
            mu = tf.matmul(w, mu)
            mu = self._relu(mu) if i<(len(self.w)-1) else mu

        vf = tf.transpose(normed_x)
        for i, v in enumerate(self.v):
            vf = tf.matmul(v, vf)
            vf = self._relu(vf) if i<(len(self.v)-1) else vf

        self.s = tf.transpose(normed_x)
        return mu, vf

    def fit(self, neighbor_id, A, g, gamma, rho, beta):
        w_new, o_new, x_new = [], [], []
        for n in range(len(self.n_nodes)):
            if n == 0:
                # Input layer
                w = self._weight_update(self.x[n], self.s, beta)
                o = self._activation_update(self.x[n+1], self.w[n+1], self.x[n], beta, gamma)
                x = self._argmin_x(o, w, self.s, beta, gamma)
                w_new.append(w)
                o_new.append(o)
                x_new.append(x)
            elif n < len(self.n_nodes) - 1:
                # Hidden layer
                w = self._weight_update(self.x[n], o, beta)
                o = self._activation_update(self.x[n+1], self.w[n+1], self.x[n], beta, gamma)
                x = self._argmin_x(o, w, o_new[-1], beta, gamma)
                w_new.append(w)
                o_new.append(o)
                x_new.append(x)
            else:
                w = self._weight_update(self.x[-1], o, beta)
                z = tf.gather(self.z, neighbor_id)[0,:]
                p = tf.gather(self.p, neighbor_id)[0,:]
                lam = self.lam
                a = self._argmin_a(g, w, o, z, p, lam, beta, rho, A)
                lam_new = self._lam_update(a, w, o, beta)
                w_new.append(w)
                x_new.append(a)

        update_op = [tf.assign(self.w[i], w_new[i]) for i in range(len(self.w))] + \
                    [tf.assign(self.o[i], o_new[i]) for i in range(len(self.o))] + \
                    [tf.assign(self.x[i], x_new[i]) for i in range(len(self.x))] + \
                    [tf.assign(self.lam, lam_new)]

        return update_op

    def fit_vf(self, value, gamma, beta):
        # vf update operation
        v_new, vo_new, vx_new = [], [], []
        for n in range(len(self.n_nodes)):
            if n == 0:
                # Input layer
                v = self._weight_update(self.vx[n], self.s, beta)
                vo = self._activation_update(self.vx[n+1], self.v[n+1], self.vx[n], beta, gamma)
                vx = self._argmin_x(vo, v, self.s, beta, gamma)
                v_new.append(v)
                vo_new.append(vo)
                vx_new.append(vx)
            elif n < len(self.n_nodes) - 1:
                # Hidden layer
                v = self._weight_update(self.vx[n], vo, beta)
                vo = self._activation_update(self.vx[n+1], self.v[n+1], self.vx[n], beta, gamma)
                vx = self._argmin_x(vo, v, vo_new[-1], beta, gamma)
                v_new.append(v)
                vo_new.append(vo)
                vx_new.append(vx)
            else:
                v = self._weight_update(self.vx[-1], vo, beta)
                vf = self._argmin_vf(tf.transpose(value), self.vlam, v, vo, beta)
                vlam_new = self.vlam + self._lam_update(vf, v, vo, beta)
                v_new.append(v)
                vx_new.append(vf)

        vf_update_op = [tf.assign(self.v[i], v_new[i]) for i in range(len(self.v))] + \
                    [tf.assign(self.vo[i], vo_new[i]) for i in range(len(self.vo))] + \
                    [tf.assign(self.vx[i], vx_new[i]) for i in range(len(self.vx))] + \
                    [tf.assign(self.vlam, vlam_new)]

        return vf_update_op

    def info_to_exchange(self, neighbor_id):
        a = self.x[-1]
        p = tf.gather(self.p, neighbor_id)[0,:]

        return a, p

    def exchange(self, sess, neighbor_id, rho):
        a, p = self.info_to_exchange(neighbor_id)
        def exchange_fn(neighbor, a_neighbor, p_neighbor, A, A_neighbor):
            z_new, p_new = self._z_and_p_update(a, a_neighbor, p, p_neighbor, A, A_neighbor, rho)
            update_op = [tf.assign(self.z[neighbor], z_new), tf.assign(self.p[neighbor], p_new)]
            sess.run(update_op, feed_dict={neighbor_id:neighbor})

        return exchange_fn

    def evaluate(self, inputs, labels):
        """
        Calculate loss
        :param inputs: inputs data
        :param outputs: ground truth
        :return: loss
        """
        _, vf = self.feedforward(inputs)
        # loss = tf.reduce_mean(tf.square(forward - labels))
        loss = tf.reduce_mean(tf.square(vf - labels))

        return loss

    # def warming(self, inputs, labels, epochs, beta, gamma, rho, A):
    #     """
    #     Warming ADMM Neural Network by minimizing sub-problems without update lambda
    #     :param inputs: input of training data samples
    #     :param outputs: label of training data samples
    #     :param epochs: number of epochs
    #     :param beta: value of beta
    #     :param gamma: value of gamma
    #     :return:
    #     """
    #     self.a0 = inputs
    #     for i in range(epochs):
    #         print("------ Warming: {:d} ------".format(i))

    #         for m in range(self.x1.shape[0]):
    #         # m = np.random.choice(A.shape[0])
    #             q = np.where(A[m] != 0)[0]
    #             # a)
    #             for n in q:
    #                 # Input layer
    #                 self.w1[n] = self._weight_update(self.z1[n], self.a0[n], beta, rho, self.y1[m,n], self.x1[m,n], A[m,n])
    #                 self.a1[n] = self._activation_update(self.w2[n], self.z2[n], self.z1[n], beta, gamma)
    #                 self.z1[n] = self._argminz(self.a1[n], self.w1[n], self.a0[n], beta, gamma)
    #                 # Hidden layer
    #                 self.w2[n] = self._weight_update(self.z2[n], self.a1[n], beta, rho, self.y2[m,n], self.x2[m,n], A[m,n])
    #                 self.a2[n] = self._activation_update(self.w3[n], self.z3[n], self.z2[n], beta, gamma)
    #                 self.z2[n] = self._argminz(self.a2[n], self.w2[n], self.a1[n], beta, gamma)
    #                 # Output layer
    #                 self.w3[n] = self._weight_update(self.z3[n], self.a2[n], beta, rho, self.y3[m,n], self.x3[m,n], A[m,n])
    #                 self.z3[n] = self._argminlastz(labels[n], self.lambda_larange[n], self.w3[n], self.a2[n], beta)
    #                 self.lambda_larange[n] = self._lambda_update(self.z3[n], self.w3[n], self.a2[n], beta)
    #             # b)
    #             v1 = 0.5 * (self.y1[m,q[0]] + self.y1[m,q[1]]) + \
    #                 0.5 * rho * (A[m,q[0]]*self.w1[q[0]] + A[m,q[1]]*self.w1[q[1]])
    #             v2 = 0.5 * (self.y2[m,q[0]] + self.y2[m,q[1]]) + \
    #                 0.5 * rho * (A[m,q[0]]*self.w2[q[0]] + A[m,q[1]]*self.w2[q[1]])
    #             v3 = 0.5 * (self.y3[m,q[0]] + self.y3[m,q[1]]) + \
    #                 0.5 * rho * (A[m,q[0]]*self.w3[q[0]] + A[m,q[1]]*self.w3[q[1]])
    #             for n in q:
    #                 self.x1[m,n] = (1.0/rho) * (self.y1[m,n] - v1) + A[m,n] * self.w1[n]
    #                 self.x2[m,n] = (1.0/rho) * (self.y2[m,n] - v2) + A[m,n] * self.w2[n]
    #                 self.x3[m,n] = (1.0/rho) * (self.y3[m,n] - v3) + A[m,n] * self.w3[n]
    #             # c)
    #                 self.y1[m,n] = v1
    #                 self.y2[m,n] = v2
    #                 self.y3[m,n] = v3


    # def drawcurve(self, train_, valid_, id, legend_1, legend_2):
    #     acc_train = np.array(train_).flatten()
    #     acc_test = np.array(valid_).flatten()

    #     plt.figure(id)
    #     plt.plot(acc_train)
    #     plt.plot(acc_test)

    #     plt.legend([legend_1, legend_2], loc='upper left')
    #     plt.draw()
    #     plt.pause(0.001)
    #     return 0
