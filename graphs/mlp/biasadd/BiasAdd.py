from graphs._AbstractGraph import AbstractGraph

import numpy as np
import tensorflow as tf


class BiasAdd(AbstractGraph):
    def __init__(self, save_dir="bias_add"):
        super(BiasAdd, self).__init__(save_dir)
        # Build the graph
        in_node = self.get_placeholder("input")
        biases = tf.Variable(tf.lin_space(1.0, 4.0, 4), name="bias")
        out_node = tf.nn.bias_add(in_node, tf.cast(biases, dtype=tf.float64), name="output")
        # Specify placeholders and output tensors
        self.graph_placeholders = [in_node]
        self.graph_output_tensors = [out_node]

    def list_inputs(self):
        return ["input"]

    def list_outputs(self):
        return ["output"]

    def _get_input(self, name):
        np.random.seed(13)
        if name == "input":
            input_0 = np.linspace(1, 40, 40).reshape(10, 4)
            return input_0

    def _get_input_shape(self, name):
        if name == "input":
            return [None, 4]


if __name__ == '__main__':
    bias_add = BiasAdd()
    bias_add.run_and_save_graph()
