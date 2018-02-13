# -*- coding: UTF-8 -*-
import gc
import sys
import json
import random
import numpy as np
import tensorflow as tf
from tfoptests import load_save_utils
from . import save_dir


if __name__ == '__main__':
    # init auto-encoder


    with tf.Session() as sess:
        arrs = []
        for i in xrange(1, 5, 1):
            arrs.append(tf.Variable(tf.constant(5, dtype=tf.float32, shape=(1, 1), name=str(str(i) + '_num'))))

        result = tf.stack(arrs, 0, name='output')
        init = tf.global_variables_initializer()
        all_saver = tf.train.Saver()
        sess.run(init)
        prediction = np.reshape(np.asarray(sess.run([result],feed_dict={})),result.shape)
        load_save_utils.save_graph(sess, all_saver, save_dir)
        load_save_utils.save_prediction(save_dir, prediction)

