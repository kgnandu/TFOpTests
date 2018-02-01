import numpy as np
import tensorflow as tf

from graphs.mathops.single_arg_ops import get_input
from helper import load_save_utils
from helper.math_ops import DifferentiableMathOps

ops = ["add"
       # , "add_n"
       , "max"
       , "min"
       , "abs"
       , "cos"
       , "acos"
       , "add"
       , "max"
       , "min"
       , "abs"
       , "ceil"
       , "min"
       # , "cross"
       , "exp"
       , "log"
       # , "log1p"
       # , "mod"
       #, "mathmul"
       # , "cumprod"
       # , "cumsum"
       # , "erf"
       # , "count_nonzero"
       # , "greater"
       # , "greater_equal"
       # , "equal"
       ]

model_name = "transform_0"
save_dir = model_name

my_feed_dict = {}
in0 = tf.placeholder("float64", [3, 3], name="input_0")
in1 = tf.placeholder("float64", [3, 3], name="input_1")
k0 = tf.Variable(tf.random_normal([3, 3]), name="in0")
my_feed_dict[in0] = get_input("input_0")
my_feed_dict[in1] = get_input("input_1")

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
    load_save_utils.save_graph(sess, all_saver, save_dir)
    load_save_utils.save_prediction(save_dir, prediction)
