import math

import numpy as np
import tensorflow as tf

from graphs.mlp.simple_ae_00 import my_input, my_output, get_input, output_data, save_dir
from helper import load_save_utils

# Autoencoder with 1 hidden layer
n_samp, n_input = get_input("input").shape
n_hidden = 2

my_feed_dict = {}
x = tf.placeholder("float", [None, n_input], name="input")
my_feed_dict[x] = get_input("input")
# Weights and biases to hidden layer
Wh = tf.Variable(tf.random_uniform((n_input, n_hidden), -1.0 / math.sqrt(n_input), 1.0 / math.sqrt(n_input)))
bh = tf.Variable(tf.zeros([n_hidden]))
h = tf.nn.tanh(tf.nn.bias_add(tf.matmul(x, Wh), bh))
# Weights and biases to hidden layer
Wo = tf.transpose(Wh)  # tied weights
# Wo = tf.Variable(tf.random_uniform((n_hidden, n_input)))
bo = tf.Variable(tf.zeros([n_input]))
y = tf.nn.tanh(tf.nn.bias_add(tf.matmul(h, Wo), bo), name="output")
# Objective functions
y_ = tf.placeholder("float", [None, n_input])
meansq = tf.reduce_mean(tf.square(y_ - y))
train_step = tf.train.GradientDescentOptimizer(0.05).minimize(meansq)

init = tf.global_variables_initializer()
all_saver = tf.train.Saver()

n_rounds = 5000
batch_size = min(50, n_samp)
'''
for i in range(n_rounds):
    sample = np.random.randint(n_samp, size=batch_size)
    batch_xs = input_data[sample][:]
    batch_ys = output_data[sample][:]
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
'''
with tf.Session() as sess:
    sess.run(init)
    for i in range(n_rounds):
        sample = np.random.randint(n_samp, size=batch_size)
        batch_xs = get_input("input")[sample][:]
        batch_ys = output_data[sample][:]
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    prediction = sess.run(y, feed_dict=my_feed_dict)
    load_save_utils.save_graph(sess, all_saver, save_dir)
    load_save_utils.save_prediction(save_dir, prediction)
