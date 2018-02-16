import numpy as np
import tensorflow as tf
from tfoptests.persistor import TensorFlowPersistor


class BiasAdd(TensorFlowPersistor):
    def _get_input(self, name):
        if name == "input":
            input_0 = np.linspace(1, 40, 40).reshape(10, 4)
            return input_0

    def _get_input_shape(self, name):
        if name == "input":
            return [None, 4]

    def _neural_net(self):
        in_node = self.get_placeholder("input")
        self.set_placeholders([in_node])
        biases = tf.Variable(tf.lin_space(1.0, 4.0, 4), name="bias")
        out_node = tf.nn.bias_add(in_node, tf.cast(biases, dtype=tf.float64), name="output")
        self.set_output_tensors([out_node])


def test_bias_add():
    # Init TFP instance
    tfp = BiasAdd(save_dir="bias_add", seed=1337)
    tfp.build_and_save_graph()


if __name__ == '__main__':
    test_bias_add()
