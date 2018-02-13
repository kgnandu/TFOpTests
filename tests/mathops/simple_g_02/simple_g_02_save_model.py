import tensorflow as tf
from tests.mathops.simple_g_02 import get_input, save_dir
from tfoptests import persistor
from tfoptests.math_ops import DifferentiableMathOps

ops = ["acos"
       ,"sin"
        , "asin"
       , "sinh"
        , "floor"
        , "asinh"
       , "min"
       , "cos"
       , "add"
       , "acosh"
       , "atan"
       , "atan2"
       , "add"
       , "elu"
       , "cosh"
       , "mod"
       , "cross"
       #, "diagpart"
       #, "diag"
       , "expm"
       ,"asinh"
       ,"atanh"
       ]

my_feed_dict = {}
in0 = tf.placeholder("float", [3, 3], name="input_0")
in1 = tf.placeholder("float", [3, 3], name="input_1")
k0 = tf.Variable(tf.random_normal([3, 3]), name="in0", dtype=tf.float32)
my_feed_dict[in0] = get_input("input_0")
my_feed_dict[in1] = get_input("input_1")

constr = DifferentiableMathOps(in0, in1)

for op in ops:
    print "Running " + op
    answer = constr.execute(op)
    print answer
    constr.set_a(answer)

finish = tf.floormod(constr.a,constr.b,name="output")

init = tf.global_variables_initializer()
all_saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    prediction = sess.run(finish, feed_dict=my_feed_dict)
    persistor.save_graph(sess, all_saver, save_dir)
    persistor.save_prediction(save_dir, prediction)

