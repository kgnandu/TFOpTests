import tensorflow as tf

# Input 0: some random rank 3 tensor
# Input 1: some random rank 2 tensor
uuu = tf.Variable(tf.random_uniform([2, 2, 2]), name="input0")
u = tf.Variable(tf.random_uniform([2, 2], name="input1"))

# Input 0: unstack to get two rank 2 tensors (unstack is tad along a dimension)
tad0, tad1 = tf.unstack(uuu, axis=0, name="tads")
# do a truncated div on one
one = tf.Variable(tf.constant(1.0, shape=[2, 2]), name="one")
ten = tf.Variable(tf.constant(10.0, shape=[2, 2]), name="ten")
allZ = tf.floordiv(tad0, one, name="floordiv0")
getTen = tf.add(allZ, ten, name="sumWithTen")

# afterScatter = tf.scatter_update(u, [1, 0], [10, 20], name="scatter")
lessThanTwo = tf.add(u, tad1)
output = tf.rsqrt(lessThanTwo, name="rsqrt")

'''
Random how to generate one hot
    YY = np.zeros((minibatch, num_classes))
    labelClass = np.random.choice(num_classes, minibatch)
    YY[np.arange(minibatch), labelClass] = 1
'''
with tf.Session() as sess:
    tf.initialize_all_variables().run()
    Z = output.eval()
    print Z
    tf.train.write_graph(sess.graph_def, "/Users/susaneraly/", "mymodel_2.txt", True)
