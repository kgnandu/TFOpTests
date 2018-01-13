import numpy as np
import tensorflow as tf

from graphs.mathops.simple_cond import get_input
from helper import load_save_utils

in0 = tf.Variable(np.linspace(1, 4, 4) + 1, name='greater')
in1 = tf.Variable(np.linspace(1, 4, 4), name='lesser')


def f1(): return in0 / 2


def f2(): return in1 / 4


def check(): return tf.reduce_sum(in0 - in1) < 2


r = tf.cond(tf.reduce_sum(in0 - in1) < 2, true_fn=lambda: f1(), false_fn=lambda: f2(),name='cond5')
r2 = tf.cond(tf.reduce_sum(in0 - in1) < 2, true_fn=lambda: f1(), false_fn=lambda: f2(),name='cond6')

last_result = tf.add(r,tf.constant(1.0,dtype=tf.float64),name='first_output_input')
last_result2 = tf.add(r2,tf.constant(1.0,dtype=tf.float64),name='second_output_input')
some_merge_result = tf.add(last_result, last_result2,name='output')


with tf.Session() as sess:
    init = tf.global_variables_initializer()
    all_saver = tf.train.Saver()

    model_name = "simple_cond"
    save_dir = model_name
    sess.run(init)
    prediction = sess.run(r,feed_dict={})

    load_save_utils.save_graph(sess, all_saver, save_dir)
    load_save_utils.save_prediction(save_dir, prediction)
