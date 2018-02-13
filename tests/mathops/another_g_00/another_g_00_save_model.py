import tensorflow as tf

from tests.mathops.another_g_00 import get_input, save_dir
from tfoptests import load_save_utils

my_feed_dict = {}
in0 = tf.placeholder("float", [4, 4, 16, 16], name="input_0")
k0 = tf.Variable(tf.random_normal([8, 8]), name="in0", dtype=tf.float32)
my_feed_dict[in0] = get_input("input_0")

n4 = tf.depth_to_space(in0, block_size=4)
n5 = tf.cumsum(n4, axis=-3, exclusive=True, reverse=True)
n6 = tf.diag_part(tf.reshape(n5,[8,8,8,8]))
n7 = tf.diag(n6)
final = tf.add(n7,k0,name="output")

init = tf.global_variables_initializer()
all_saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    prediction = sess.run(final, feed_dict=my_feed_dict)
    load_save_utils.save_graph(sess, all_saver, save_dir)
    load_save_utils.save_prediction(save_dir, prediction)
