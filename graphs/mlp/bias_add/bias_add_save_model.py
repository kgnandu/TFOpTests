from __future__ import print_function

import tensorflow as tf

from graphs.mlp.bias_add import get_input, get_tf_persistor

persistor = get_tf_persistor()

# tf Graph input
my_feed_dict = {}
in_0 = tf.placeholder("float", [None, 4], name="input")
my_feed_dict[in_0] = get_input("input")
biases = tf.Variable(tf.lin_space(1.0, 4.0, 4), name="bias")
output = tf.nn.bias_add(in_0, biases, name="output")

init = tf.global_variables_initializer()
all_saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    prediction = sess.run(output, feed_dict=my_feed_dict)
    persistor.save_prediction(prediction)
    persistor.save_graph(sess, all_saver)
