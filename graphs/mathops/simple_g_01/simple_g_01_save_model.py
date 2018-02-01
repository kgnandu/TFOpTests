import numpy as np
import tensorflow as tf

from graphs.mathops.simple_g_01 import get_input, save_dir
from helper import load_save_utils

my_feed_dict = {}
in0 = tf.placeholder("float64", [3, 3], name="input_0")
in1 = tf.placeholder("float64", [3, 3], name="input_1")
my_feed_dict[in0] = get_input("input_0")
my_feed_dict[in1] = get_input("input_1")

n0 = tf.add(np.arange(-4., 5., 1.).astype(np.float64).reshape(3, 3), in0)
n1 = tf.abs(n0)
n2 = tf.rsqrt(n1)
n3 = tf.add(n1, tf.Variable(tf.random_normal([3, 3],dtype=tf.float64)))
n4 = tf.floordiv(n3, in1)
output = tf.tanh(n4, name="output")

init = tf.global_variables_initializer()
all_saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    prediction = sess.run(output, feed_dict=my_feed_dict)
    print prediction
    load_save_utils.save_graph(sess, all_saver, save_dir)
    load_save_utils.save_prediction(save_dir, prediction)
