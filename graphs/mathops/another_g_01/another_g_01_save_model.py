import tensorflow as tf
from graphs.mathops.another_g_01 import get_input, save_dir
from helper import load_save_utils

my_feed_dict = {}
in1 = tf.placeholder("float", [16, 16], name="input_1")
in2 = tf.placeholder("float", [16, 16], name="input_2")
k0 = tf.Variable(tf.random_normal([8, 1, 8]), name="in0", dtype=tf.float32)
my_feed_dict[in1] = get_input("input_1")
my_feed_dict[in2] = get_input("input_2")

n1 = tf.concat([in1, in2], axis=-2)
n3 = tf.reshape(n1, [8, 8, 8])
n4 = tf.pow(n3,n3)
n5 = tf.tan(n4)
n6 = tf.negative(n5)
n7 = tf.multiply(n6,n4)
final = tf.subtract(n7,k0,name="output")


init = tf.global_variables_initializer()
all_saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    prediction = sess.run(final, feed_dict=my_feed_dict)
    load_save_utils.save_graph(sess, all_saver, save_dir)
    load_save_utils.save_prediction(save_dir, prediction)