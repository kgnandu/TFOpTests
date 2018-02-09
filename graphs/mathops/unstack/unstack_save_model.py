import numpy as np
import tensorflow as tf

from graphs.mathops.unstack import save_dir
from helper import load_save_utils

if __name__ == '__main__':

    with tf.Session() as sess:
        arrs = tf.Variable(tf.constant(np.reshape(np.linspace(1, 25, 25), (5, 5))))
        result = tf.unstack(arrs, axis=0, name='output')
        init = tf.global_variables_initializer()
        all_saver = tf.train.Saver()
        sess.run(init)
        prediction = sess.run(result, feed_dict={})
        load_save_utils.save_graph(sess, all_saver, save_dir)
        print prediction
        load_save_utils.save_predictions(save_dir,prediction)
