import numpy as np
import tensorflow as tf

from tests.mathops.add_n import save_dir, get_input
from tfoptests import load_save_utils
from tfoptests.math_ops import DifferentiableMathOps

ops = ["add"
    , "add_n"]

my_feed_dict = {}
in0 = tf.placeholder("float", [3, 3], name="input_0")
my_feed_dict[in0] = get_input("input_0")
in1 = tf.placeholder("float", [3, 3], name="input_1")
my_feed_dict[in1] = get_input("input_1")
k0 = tf.Variable(tf.random_normal([3, 3]), name="in0", dtype=tf.float32)

constr = DifferentiableMathOps(in0, in1)

for op in ops:
    print "Running " + op
    answer = constr.execute(op)
    print answer
    constr.set_a(answer)

finish = tf.rsqrt(answer, name="output")

init = tf.global_variables_initializer()
all_saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    prediction = sess.run(finish, feed_dict=my_feed_dict)
    print prediction
    load_save_utils.save_graph(sess, all_saver, save_dir)
    load_save_utils.save_prediction(save_dir, prediction)
