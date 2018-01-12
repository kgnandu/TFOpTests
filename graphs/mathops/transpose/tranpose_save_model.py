import numpy as np
import tensorflow as tf

from graphs.mathops.transpose import get_input, save_dir
from helper import load_save_utils

np.random.seed(13)

my_feed_dict = {}
in0 = tf.placeholder("float", [3, 3], name="input_0")
my_feed_dict[in0] = get_input("input_0")

k0 = tf.Variable(tf.random_normal([3, 3]), name="k0", dtype=tf.float32)
in1 = tf.transpose(in0, name="input_1")
finish = tf.add(in1, k0, name="output")

init = tf.global_variables_initializer()
all_saver = tf.train.Saver()

with tf.Session() as sess:
    input_0 = get_input("input_0")
    sess.run(init)
    prediction = sess.run(finish, feed_dict=my_feed_dict)
    load_save_utils.save_graph(sess, all_saver, save_dir)
    load_save_utils.save_prediction(save_dir, prediction)


