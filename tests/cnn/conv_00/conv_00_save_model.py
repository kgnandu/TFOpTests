import tensorflow as tf

from tests.mlp.bias_add import BaseCNNInput, get_tf_persistor
from tfoptests.nn_image_ops import NNImageOps

persistor = get_tf_persistor()
input = BaseCNNInput()

my_feed_dict = {}
in0 = tf.placeholder("float", input.imsize, name="input_0")
my_feed_dict[in0] = input.get_input("input_0")

constr = NNImageOps(in0)

constr.set_image(in0)
constr.set_filter_hw_inout(5, 5, 3, 3)
constr.set_kernel_hw(3, 3)
constr.set_stride_hw(3, 3)

in1 = constr.execute("conv2d")
constr.set_image(in1)

in2 = constr.execute("avg_pool")
constr.set_image(in2)

in3 = constr.execute("conv2d")
constr.set_image(in3)

in4 = constr.execute("max_pool")

in5 = constr.flatten_convolution(in4)

finish = tf.matmul(in5, tf.Variable(tf.random_uniform([3, 2])), name="output")  # calc required dims by hand

init = tf.global_variables_initializer()
all_saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    prediction = sess.run(finish, feed_dict=my_feed_dict)
    persistor.save_graph(sess, all_saver)
    persistor.save_prediction(prediction)
