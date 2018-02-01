import tensorflow as tf

from graphs.mathops.node_with_multipleouts import get_input, save_dir
from helper import load_save_utils

my_feed_dict = {}
in0 = tf.placeholder("float64", [2, 3, 4, 5], name="input_0")
my_feed_dict[in0] = get_input("input_0")

unstacked = tf.unstack(in0, axis=-2)
unstack1 = unstacked[0]
unstack2 = unstacked[1]  # 2x3x5 now

n1 = unstack1 + tf.Variable(tf.zeros([2, 3, 5],dtype=tf.float64))
n2 = unstack2

output = tf.stack([n1, n2, unstacked[2], unstacked[3]], axis=-4, name="output")

init = tf.global_variables_initializer()
all_saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    prediction = sess.run(output, feed_dict=my_feed_dict)
    print prediction
    load_save_utils.save_graph(sess, all_saver, save_dir)
    load_save_utils.save_prediction(save_dir, prediction)

    print("===========")
    print(sess.run(n1, feed_dict=my_feed_dict))
