import tensorflow as tf

from graphs.mathops.non2d_1 import get_input, save_dir
from helper import load_save_utils

my_feed_dict = {}
in0 = tf.placeholder("float64", [], name="scalar")
my_feed_dict[in0] = get_input("scalar")

in1 = tf.placeholder("float64", [2], name="vector")
my_feed_dict[in1] = get_input("vector")
k0 = tf.Variable(tf.random_normal([2, 1],dtype=tf.float64), name="someweight")

i0 = tf.reshape(tf.reduce_sum(in1), [])
i1 = in0 + in1
final = tf.matmul(tf.expand_dims(in1, 0), k0, name="output")

init = tf.global_variables_initializer()
all_saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    prediction = sess.run(final, feed_dict=my_feed_dict)
    print prediction
    print prediction.shape
    load_save_utils.save_graph(sess, all_saver, save_dir)
    load_save_utils.save_prediction(save_dir, prediction)
