from __future__ import print_function
import numpy as np
import tensorflow as tf

from tfoptests import load_save_utils

base_dir = "/Users/susaneraly/SKYMIND/nd4j/nd4j-backends/nd4j-tests/src/test/resources/tf_graphs/examples"
model_name = "conv_0"
save_dir = base_dir + "/" + model_name

imsize = [4, 28, 28, 3]
in0 = tf.placeholder("float", imsize, name="input_0")

filter = tf.Variable(tf.random_normal([5, 5, 3, 3]), name="filterW", dtype=tf.float32)
finish = tf.nn.conv2d(in0, filter, [1, 3, 3, 1], padding='SAME', name="output")

init = tf.global_variables_initializer()
all_saver = tf.train.Saver()

with tf.Session() as sess:
    np.random.seed(42)
    input_0 = np.random.uniform(size=imsize)
    print(input_0)
    # This writes input as csv so I can read into nd4j
    load_save_utils.save_input(input_0, "input_0", save_dir)  # change this to take a list of inputs for later...

    sess.run(init)
    prediction = sess.run(finish, feed_dict={in0: input_0})
    print(prediction)
    print(prediction.shape)
    tf.train.write_graph(sess.graph_def, "/Users/susaneraly/", "mymodel_2.txt", True)
    # All the below is to save the graph and variables etc to pb
    load_save_utils.save_graph(sess, all_saver, save_dir)
    load_save_utils.freeze_n_save_graph(save_dir)
    load_save_utils.save_prediction(save_dir, prediction)
