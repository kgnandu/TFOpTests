import tensorflow as tf

from helper import load_save_utils
from graphs.mathops.non2d_2 import get_input, save_dir

my_feed_dict = {}
in0 = tf.placeholder("float64", [1, 2], name="rank2dF")  # [1,2]
my_feed_dict[in0] = get_input("rank2dF")

in1 = tf.placeholder("float64", [2, 1], name="rank2dB")
my_feed_dict[in1] = get_input("rank2dB")

in2 = tf.placeholder("float64", [1, 3, 2], name="rank3d")
my_feed_dict[in2] = get_input("rank3d")

in3 = tf.placeholder("float64", [3, 1, 2], name="rank3dB")
my_feed_dict[in3] = get_input("rank3dB")

i0 = tf.squeeze(in0)
k0 = tf.Variable(get_input("rank2dB"), name="someweight", dtype=tf.float64)

i1 = tf.stack([in0, tf.transpose(in1)])
i2 = tf.multiply(i1, i0)  # how is this 2,1,2
i3 = tf.tile(i1, [1, 3, 1])  # 2,3,2
i4 = tf.transpose(in2) + i3  # 2,3,2
i5 = tf.reduce_sum(i4)  # now a scalar

i6 = tf.squeeze(in2 + i5)  # 3,2
i7 = in3 + i6  # 3,3,2

i8 = tf.unstack(k0, axis=0)[0]  # vector (1,)
i9 = tf.unstack(k0, axis=1)[0]  # vector (2,)
i10 = i9 + i8 + i7

final = tf.identity(i10, name="output")

init = tf.global_variables_initializer()
all_saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    prediction = sess.run(final, feed_dict=my_feed_dict)
    print prediction
    print prediction.shape
    load_save_utils.save_graph(sess, all_saver, save_dir)
    load_save_utils.save_prediction(save_dir, prediction)
