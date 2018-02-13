import tensorflow as tf

from tests.mathops.non2d_0 import get_input, save_dir
from tfoptests import persistor

my_feed_dict = {}
in0 = tf.placeholder("float", [], name="scalar")
my_feed_dict[in0] = get_input("scalar")
k0 = tf.Variable(tf.random_normal([2, 1]), name="someweight", dtype=tf.float32)
a = tf.reduce_sum(in0 + k0) #gives a scalar
final = tf.reduce_sum(a + k0, name="output",axis=0) #gives a vector

init = tf.global_variables_initializer()
all_saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    prediction = sess.run(final, feed_dict=my_feed_dict)
    print prediction
    print prediction.shape
    persistor.save_graph(sess, all_saver, save_dir)
    persistor.save_prediction(save_dir, prediction)
