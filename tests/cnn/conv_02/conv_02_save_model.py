import tensorflow as tf
from tests.cnn.conv_02 import save_dir, imsize, get_input
from tfoptests import persistor

my_feed_dict = {}
in0 = tf.placeholder("float", imsize, name="input_0")
my_feed_dict[in0] = get_input("input_0")

'''
atrous_conv2d(
    value,
    filters,
    rate,
    padding,
    name=None
)
value: A 4-D Tensor of type float. It needs to be in the default "NHWC" format. Its shape is [batch, in_height, in_width, in_channels].
filters: A 4-D Tensor with the same type as value and shape [filter_height, filter_width, in_channels, out_channels]. filters' in_channels dimension must match that of value. Atrous convolution is equivalent to standard convolution with upsampled filters with effective height filter_height + (filter_height - 1) * (rate - 1) and effective width filter_width + (filter_width - 1) * (rate - 1), produced by inserting rate - 1 zeros along consecutive elements across the filters' spatial dimensions.
rate: A positive int32. The stride with which we sample input values across the height and width dimensions. Equivalently, the rate by which we upsample the filter values by inserting zeros across the height and width dimensions. In the literature, the same parameter is sometimes called input stride or dilation.
padding: A string, either 'VALID' or 'SAME'. The padding algorithm.
name: Optional name for the returned tensor.
'''

# [filter_height, filter_width, in_channels, out_channels]. in_channels must match between input and filter.
filter_one = tf.Variable(tf.random_uniform([4, 5, imsize[-1], imsize[-1] / 2]), name="filter1")

atrous_one = tf.nn.atrous_conv2d(in0, filters=filter_one, rate=8, padding='SAME', name="atrous_one")

filter_two = tf.Variable(tf.random_uniform([31, 31, 2, 1]), name="filter2")
finish = tf.nn.atrous_conv2d(atrous_one, filters=filter_two, rate=2, padding='VALID', name="output")

init = tf.global_variables_initializer()
all_saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    prediction = sess.run(finish, feed_dict=my_feed_dict)
    print (prediction.shape)
    persistor.save_graph(sess, all_saver, save_dir)
    persistor.save_prediction(save_dir, prediction)
