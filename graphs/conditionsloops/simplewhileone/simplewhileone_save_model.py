import tensorflow as tf

from graphs.conditionsloops.simplewhileone import get_input, save_dir
from helper import load_save_utils

my_feed_dict = {}
in0 = tf.placeholder("float", [2, 2], name="input_0")
in1 = tf.placeholder("float", [], name="input_1")
my_feed_dict[in0] = get_input("input_0")
my_feed_dict[in1] = get_input("input_1")

in3 = tf.Variable(tf.constant(2.0, shape=[], name="addVal"))


def body(x0, x1):
    return [tf.add(x0, in3), tf.subtract(x1, 1)]


def condition(x0, x1):
    return tf.less(tf.reduce_sum(x0), x1)


in0p, in1p = tf.while_loop(condition, body, [in0, in1])
inp = tf.add(in0p, in1p)
output = tf.identity(inp, name="output")

init = tf.global_variables_initializer()
all_saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    prediction = sess.run(output, feed_dict=my_feed_dict)
    print prediction
    load_save_utils.save_graph(sess, all_saver, save_dir)
    load_save_utils.save_prediction(save_dir, prediction)
