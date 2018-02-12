from __future__ import print_function

import tensorflow as tf
import numpy as np

in_0 = tf.placeholder("float", [None, 4], name="input")
biases = tf.Variable(tf.lin_space(1.0, 4.0, 4), name="bias")
output = tf.nn.bias_add(in_0, biases, name="output")

init = tf.global_variables_initializer()
all_saver = tf.train.Saver()
with tf.Session() as sess:
    input_0 = np.linspace(1, 40, 40).reshape(10, 4)
    sess.run(init)
    prediction = sess.run(output, feed_dict={in_0: input_0})
    print(prediction)
