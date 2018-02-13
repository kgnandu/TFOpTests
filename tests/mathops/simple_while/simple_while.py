import numpy as np
import tensorflow as tf

from tests.mathops.simple_cond import get_input
from tfoptests import persistor

i1 = tf.Variable(tf.constant(0),name='loop_var')
c = lambda i: tf.less(i, 10)
b = lambda i: tf.add(i, 1)
r = tf.while_loop(c, b, [i1],name='output')




with tf.Session() as sess:
    init = tf.global_variables_initializer()
    all_saver = tf.train.Saver()

    model_name = "simple_while"
    save_dir = model_name
    sess.run(init)
    prediction = sess.run(r,feed_dict={})

    persistor.save_graph(sess, all_saver, save_dir)
    persistor.save_prediction(save_dir, prediction)
