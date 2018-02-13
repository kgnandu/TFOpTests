import tensorflow as tf

from tests.cnn.pool_test_1 import save_dir, imsize, get_input
from tfoptests import persistor
from tfoptests.nn_image_ops import NNImageOps

my_feed_dict = {}
in0 = tf.placeholder("float", imsize, name="input_0")
my_feed_dict[in0] = get_input("input_0")

constr = NNImageOps(in0)

constr.set_image(in0)
constr.set_filter_hw_inout(2, 2, 2, 2)
constr.set_kernel_hw(2, 2)
constr.set_stride_hw(1, 1)


in1 = constr.execute("max_pool")
constr.set_image(in1)

finish = tf.identity(in1, name="output")  # calc required dims by hand
k = tf.Variable(tf.random_uniform([3, 2]))

init = tf.global_variables_initializer()
all_saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    prediction = sess.run(finish, feed_dict=my_feed_dict)
    persistor.save_graph(sess, all_saver, save_dir)
    persistor.save_prediction(save_dir, prediction)