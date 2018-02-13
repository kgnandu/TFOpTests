import tensorflow as tf

from tests.mathops.expand_dims import get_input, save_dir
from tfoptests import persistor

my_feed_dict = {}
in0 = tf.placeholder("float", [3, 4], name="input_0")
my_feed_dict[in0] = get_input("input_0")
k0 = tf.Variable(tf.random_normal([3, 1, 4]), name="in0", dtype=tf.float32)

in0_expanded = tf.expand_dims(in0,axis=-2)
finish = tf.add(in0_expanded,k0,name="output")

init = tf.global_variables_initializer()
all_saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    prediction = sess.run(finish, feed_dict=my_feed_dict)
    print prediction
    persistor.save_graph(sess, all_saver, save_dir)
    persistor.save_prediction(save_dir, prediction)