import tensorflow as tf
from tensorflow.contrib import rnn
from tests.mlp.bias_add import SimpleLstmInput, get_tf_persistor

persistor = get_tf_persistor()
inputs = SimpleLstmInput()

num_hidden = 3
learning_rate = 0.001
training_steps = 10000
display_step = 10

# tf Graph input
X = tf.placeholder("float", [None, input.featuresize, input.timesteps - 1], name="input")
Y = tf.placeholder("float", [None, input.featuresize])
weights = {
    'out': tf.Variable(tf.random_normal([num_hidden, input.featuresize]))
}
biases = {
    'out': tf.Variable(tf.random_normal([input.featuresize]))
}


def RNN(x, weights, biases):
    x = tf.unstack(x, axis=2)
    lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
    # inputs: A length T list of inputs,
    #  each a Tensor of shape [batch_size, input_size], or a nested tuple of such elements.
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
    return tf.matmul(outputs[-1], weights['out']) + biases['out']


preout = RNN(X, weights, biases)
output = tf.identity(preout, name="output")

# Define loss and optimizer
loss_op = tf.reduce_sum(tf.pow(output - Y, 2)) / (2 * input.minibatch)  # MSE
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)
accuracy = tf.pow(output - Y, 2)

all_saver = tf.train.Saver()
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:
    # Run the initializer
    sess.run(init)
    for step in range(1, training_steps + 1):
        sess.run(train_op, feed_dict={X: input.get_input("input"), Y: input.get_output("output")})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: input.get_input("input"),
                                                                 Y: input.get_output("output")})
            print("Step " + str(step) + ", Loss= " +
                  "{:.4f}".format(loss))
            # print(acc)
    print("Optimization Finished!")

    prediction = sess.run(output, feed_dict={X: input.get_input("input")})
    print(input.get_input("input"))
    print(input.get_input("input").shape)
    print("====")
    print(input.get_output("output"))
    print("====")
    print(prediction)
    persistor.save_prediction(prediction)
    persistor.save_graph(sess, all_saver)
