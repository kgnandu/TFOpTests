import numpy as np
import math
import tensorflow as tf
from tfoptests.persistor import TensorFlowPersistor


class SimpleAE(TensorFlowPersistor):
    def __init__(self, *args, **kwargs):
        super(SimpleAE, self).__init__(*args, **kwargs)
        self.train_input, self.train_output = generate_input_output()
        print(self.train_input)
        print(self.train_output)
        self.n_input = self.train_input.shape[1]
        self.n_hidden = 2

    def _get_input(self, name):
        if name == "input":
            return self.train_input

    def _get_input_shape(self, name):
        if name == "input":
            return [None, self.n_input]

    def _neural_net(self):
        # Set input tensor
        expected_out = tf.placeholder(dtype=tf.float64, shape=(None, self.n_input), name="target")
        in_node = self.get_placeholder("input")
        self.set_placeholders([in_node])
        # Build Net
        # Hidden layer
        Wh = tf.Variable(tf.random_uniform((self.n_input, self.n_hidden), -1.0 / math.sqrt(self.n_input),
                                           1.0 / math.sqrt(self.n_input), dtype=tf.float64))
        bh = tf.Variable(tf.random_normal([self.n_hidden], dtype=tf.float64))
        h = tf.nn.tanh(tf.nn.bias_add(tf.matmul(in_node, Wh), bh))
        # Output layer
        Wo = tf.transpose(Wh)  # tied weights
        bo = tf.Variable(tf.zeros([self.n_input], dtype=tf.float64))
        out_node = tf.nn.tanh(tf.nn.bias_add(tf.matmul(h, Wo), bo), name="output")
        self.set_output_tensors([out_node])
        meansq = tf.reduce_mean(tf.square(expected_out - out_node))
        train_step = tf.train.GradientDescentOptimizer(0.05).minimize(meansq)
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        for i in range(10):
            sess.run(train_step, feed_dict={in_node: self.train_input, expected_out: self.train_output})
        out_after_train = sess.run(self.graph_output_tensors, feed_dict={in_node: self.train_input})
        return sess, dict(zip(self._list_output_node_names(), out_after_train))


def generate_input_output():
    np.random.seed(13)
    my_input = np.array([[2.0, 1.0, 1.0, 2.0],
                         [-2.0, 1.0, -1.0, 2.0],
                         [0.0, 1.0, 0.0, 2.0],
                         [0.0, -1.0, 0.0, -2.0],
                         [0.0, -1.0, 0.0, -2.0]])

    my_output = np.array([[2.0, 1.0, 1.0, 2.0],
                          [-2.0, 1.0, -1.0, 2.0],
                          [0.0, 1.0, 0.0, 2.0],
                          [0.0, -1.0, 0.0, -2.0],
                          [0.0, -1.0, 0.0, -2.0]])
    noisy_input = my_input + .2 * np.random.random_sample((my_input.shape)) - .1
    # Scale to [0,1]
    scaled_input_1 = np.divide((noisy_input - noisy_input.min()), (noisy_input.max() - noisy_input.min()))
    scaled_output_1 = np.divide((my_output - my_output.min()), (my_output.max() - my_output.min()))
    # Scale to [-1,1]
    input_data = (scaled_input_1 * 2) - 1
    output_data = (scaled_output_1 * 2) - 1
    return input_data, output_data


def test_simple_ae():
    # Init TFP instance
    tfp = SimpleAE(save_dir="ae_00", seed=1337)
    tfp.build_and_save_graph()


if __name__ == '__main__':
    test_simple_ae()
