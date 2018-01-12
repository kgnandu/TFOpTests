import tensorflow as tf

from graphs.mathops.non2d_0A import get_input,save_dir
from helper import load_save_utils

my_feed_dict = {}
in0 = tf.placeholder(name="scalarA",dtype=tf.int32)
my_feed_dict[in0] = get_input("scalarA")
in1 = tf.placeholder(name="scalarB", dtype=tf.int32)
my_feed_dict[in1] = get_input("scalarB")

some_vector = tf.stack([in0, in1]) #[2,] shape with value [5,2]
i0 = tf.Variable(get_input("some_weight"), dtype=tf.float32) #shape [3,4]
final = tf.tile(i0,some_vector,name="output")

init = tf.global_variables_initializer()
all_saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    prediction = sess.run(final, feed_dict=my_feed_dict)
    print prediction
    print prediction.shape
    load_save_utils.save_graph(sess, all_saver, save_dir)
    load_save_utils.save_prediction(save_dir, prediction)
