import tensorflow as tf

from graphs.cnn.conv_01 import imsize, get_input, save_dir
from helper import load_save_utils
from helper.nn_image_ops import NNImageOps

my_feed_dict = {}
in0 = tf.placeholder("float", imsize, name="input_0")
my_feed_dict[in0] = get_input("input_0")

constr = NNImageOps(in0)

constr.set_image(in0)
#[filter_depth, filter_height, filter_width, in_channels, out_channels]. in_channels must match between input and filter.
filter = [2, 5, 5, 3, 4]
constr.set_filter(filter)
#Must have strides[0] = strides[4] = 1.
stride = [1, 5, 4, 3, 1]
constr.set_stride(stride)

in1 = constr.execute("conv3d")
constr.set_image(in1)

in2 = constr.flatten_convolution(in1)

finish = tf.matmul(in2, tf.Variable(tf.random_uniform([280, 3])), name="output")  # calc required dims by hand

init = tf.global_variables_initializer()
all_saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    prediction = sess.run(finish, feed_dict=my_feed_dict)
    load_save_utils.save_graph(sess, all_saver, save_dir)
    load_save_utils.save_prediction(save_dir, prediction)