import tensorflow as tf

from tests.cnn.conv_03 import imsize, get_input, save_dir
from tfoptests import load_save_utils

my_feed_dict = {}
in0 = tf.placeholder("float", imsize, name="input_0")
my_feed_dict[in0] = get_input("input_0")

'''
dilation2d(
    input,
    filter,
    strides,
    rates,
    padding,
    name=None
)
value: A 4-D Tensor of type float. It needs to be in the default "NHWC" format. Its shape is [batch, in_height, in_width, in_channels].
filters:  3-D with shape [filter_height, filter_width, depth].
strides: list of ints that has length >= 4. The stride of the sliding window for each dimension of the input tensor. Must be: [1, stride_height, stride_width, 1].
rate:A list of ints that has length >= 4. The input stride for atrous morphological dilation. Must be: [1, rate_height, rate_width, 1].
padding: A string, either 'VALID' or 'SAME'. The padding algorithm.
name: Optional name for the returned tensor.
'''

# [filter_height, filter_width, in_channels, out_channels]. in_channels must match between input and filter.
filter_one = tf.Variable(tf.random_uniform([4, 5, imsize[-1]]), name="filter1")

atrous_one = tf.nn.dilation2d(in0, filter=filter_one, strides=[1, 2, 3, 1], rates=[1, 5, 7, 1], padding='SAME',
                              name="atrous_one")

filter_two = tf.Variable(tf.random_uniform([11, 7, 4]), name="filter2")
finish = tf.nn.dilation2d(atrous_one, filter=filter_two, strides=[1, 3, 2, 1], rates=[1, 2, 3, 1], padding='VALID',
                              name="output")

init = tf.global_variables_initializer()
all_saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    prediction = sess.run(finish, feed_dict=my_feed_dict)
    print (prediction.shape)
    load_save_utils.save_graph(sess, all_saver, save_dir)
    load_save_utils.save_prediction(save_dir, prediction)
