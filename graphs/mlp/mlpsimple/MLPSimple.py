from graphs._AbstractGraph import AbstractGraph

import numpy as np
import tensorflow as tf

n_hidden_1 = 10
num_input = 5
num_classes = 3
mini_batch = 4


class MLPSimple(AbstractGraph):
    def __init__(self, save_dir="mlp_00"):
        super(MLPSimple, self).__init__(save_dir)
        # Build the graph
        weights = {
            'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1], dtype=tf.float64), name="l0W"),
            'out': tf.Variable(tf.random_normal([n_hidden_1, num_classes], dtype=tf.float64), name="l1W")
        }
        biases = {
            'b1': tf.Variable(tf.random_normal([n_hidden_1], dtype=tf.float64), name="l0B"),
            'out': tf.Variable(tf.random_normal([num_classes], dtype=tf.float64), name="l1B")
        }
        in_node = self.get_placeholder("input")
        layer_1 = tf.nn.bias_add(tf.matmul(in_node, weights['h1']), biases['b1'], name="l0Preout")
        layer_1_post_actv = tf.abs(layer_1, name="l0Out")
        out_layer = tf.nn.bias_add(tf.matmul(layer_1_post_actv, weights['out']), biases['out'], name="l1PreOut")
        out_node = tf.nn.softmax(out_layer, name='output')
        # Specify placeholders and output tensors
        self.graph_placeholders = [in_node]
        self.graph_output_tensors = [out_node]

    def _get_input(self, name):
        np.random.seed(13)
        if name == "input":
            input_0 = np.random.uniform(size=(mini_batch, num_input))
            return input_0

    def _get_input_shape(self, name):
        if name == "input":
            return [None, num_input]


if __name__ == '__main__':
    mlp_simple = MLPSimple()
    mlp_simple.run_and_save_graph()
