import tensorflow as tf
import numpy as np


def body(x, y, z):
    x += 25
    y += 7
    return (x, y, z)


def condition(x, y, z):
    return tf.reduce_sum(x) < 400


f = tf.Variable(tf.constant(0, shape=[2, 2]), name="phi")
a = tf.Variable(tf.constant(0, shape=[2, 2]), name="alpha")
z = tf.Variable(tf.constant(0, shape=[2, 2]), name="omega")

with tf.Session() as sess:
    tf.initialize_all_variables().run()
    alpha, phi, zeta = tf.while_loop(condition, body, [a, f, z])
    alpha += 10
    phi += 5
    sess.run(tf.initialize_all_variables())
    print alpha.eval()
    print phi.eval()
    tf.train.write_graph(sess.graph_def, "/Users/susaneraly/", "mymodel_2.txt", True)
