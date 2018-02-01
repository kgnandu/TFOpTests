from __future__ import print_function

import numpy as np
import tensorflow as tf

from graphs.mlp.bias_add import save_dir, get_input
from helper import load_save_utils

# tf Graph input
my_feed_dict = {}
in_0 = tf.placeholder("float64", [None, 4], name="input")
my_feed_dict[in_0] = get_input("input")
biases = tf.Variable(tf.lin_space(1.0, 4.0, 4), name="bias")
output = tf.nn.bias_add(in_0, tf.cast(biases, dtype=tf.float64), name="output")

init = tf.global_variables_initializer()
all_saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    prediction = sess.run(output, feed_dict=my_feed_dict)
    load_save_utils.save_prediction(save_dir, prediction)
    load_save_utils.save_graph(sess, all_saver, save_dir)
