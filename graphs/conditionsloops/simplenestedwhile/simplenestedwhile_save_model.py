import tensorflow as tf

from graphs.conditionsloops.simplenestedwhile import get_input, save_dir
from helper import load_save_utils

my_feed_dict = {}
in0 = tf.placeholder("float", [2, 2], name="input_0")
in1 = tf.placeholder("float", [3, 3], name="input_1")
my_feed_dict[in0] = get_input("input_0")
my_feed_dict[in1] = get_input("input_1")

c0 = tf.Variable(tf.constant(10.0, shape=[], name="addVal0"))
c1 = tf.Variable(tf.constant(2.0, shape=[], name="addVal1"))
c2 = tf.Variable(tf.constant(1.0, shape=[], name="addVal2"))


def outer_body_cond(x0, x1):
    return tf.less(tf.reduce_mean(x0), c0)


def outer_body_fn(x0, x1):
    def inner_body_cond(x0, x1):
        return tf.less_equal(tf.reduce_sum(x1), tf.reduce_sum(x0))

    def inner_body_fn(x0, x1):
        x1 = tf.add(x1, c2)
        return [x0, x1]

    x0 = tf.add(x0, c1)
    x0, x1 = tf.while_loop(inner_body_cond, inner_body_fn, [x0, x1])
    x1 = tf.subtract(x1, c2)
    return [x0, x1]


in0p, in1p = tf.while_loop(outer_body_cond, outer_body_fn, [in0, in1])
inp = tf.add(in0p, tf.reduce_mean(in1p))
output = tf.identity(inp, name="output")

init = tf.global_variables_initializer()
all_saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    #A, C = sess.run([in0p, in1p], feed_dict=my_feed_dict)
    #print A
    #print C
    prediction = sess.run(output, feed_dict=my_feed_dict)
    print prediction
    load_save_utils.save_graph(sess, all_saver, save_dir)
    load_save_utils.save_prediction(save_dir, prediction)
