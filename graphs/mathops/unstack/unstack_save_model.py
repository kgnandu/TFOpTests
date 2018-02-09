import numpy as np
import tensorflow as tf
from itertools import izip


from graphs.mathops.unstack import save_dir
from helper import load_save_utils

if __name__ == '__main__':
    with tf.Session() as sess:
        arrs = tf.Variable(tf.constant(np.reshape(np.linspace(1, 25, 25), (5, 5))))
        result = tf.unstack(arrs, axis=0, name='outputs')
        r0 = tf.identity(result[0], name="out0")
        r1 = tf.identity(result[1], name="out1")
        r2 = tf.identity(result[2], name="out2")
        init = tf.global_variables_initializer()
        all_saver = tf.train.Saver()
        sess.run(init)
        predictions = sess.run([r0, r1, r2], feed_dict={})
        print predictions
        load_save_utils.save_graph(sess, all_saver, save_dir)
        load_save_utils.save_predictions(save_dir, dict(izip(["out0", "out1", "out2"], predictions)))
