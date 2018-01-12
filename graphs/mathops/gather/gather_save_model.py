import tensorflow as tf

from graphs.mathops.gather import get_input, save_dir
from helper import load_save_utils

my_feed_dict = {}
in0 = tf.placeholder("float", [2, 4, 3, 2], name="input_1")
in1 = tf.placeholder("float", [3, 2], name="input_2")
k0 = tf.Variable(tf.random_normal([3, 2]), name="in0", dtype=tf.float32)
my_feed_dict[in0] = get_input("input_1")
my_feed_dict[in1] = get_input("input_2")

n0 = tf.gather(in0,[1,0], axis=-2) #2,4,2,2
n1 = tf.gather_nd(n0,[[0,2,1],[0,1,0],[1,3,1]]) # 3,2
final = tf.stack([n1,k0,in1],axis=-1,name="output") #3, 2, 2
init = tf.global_variables_initializer()
all_saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    prediction = sess.run(final, feed_dict=my_feed_dict)
    print prediction
    load_save_utils.save_graph(sess, all_saver, save_dir)
    load_save_utils.save_prediction(save_dir, prediction)