import numpy as np
import tensorflow as tf

from graphs.mathops.simple_g_00 import save_dir
from helper import load_save_utils

in0 = tf.Variable(tf.random_normal([3, 3]), name="in0", dtype=tf.float32)
n0 = tf.add(np.arange(-4., 5., 1.).astype(np.float32).reshape(3, 3), in0)
n1 = tf.abs(n0)
n2 = tf.rsqrt(n1)
output = tf.tanh(n2, name="output")

init = tf.global_variables_initializer()
all_saver = tf.train.Saver()

with tf.Session() as sess:
    np.random.seed(13)
    sess.run(init)
    prediction = sess.run(output)
    load_save_utils.save_graph(sess, all_saver, save_dir)
    load_save_utils.save_prediction(save_dir, prediction)
