import numpy as np
import tensorflow as tf
from tfoptests.persistor import TensorFlowPersistor

n_hidden_1 = 10
num_input = 5
mini_batch = 4
num_classes = 3


class VanillaMLP(TensorFlowPersistor):
    def _get_input(self, name):
        if name == "input":
            input_0 = np.random.uniform(size=(mini_batch, num_input))
            return input_0

    def _get_input_shape(self, name):
        if name == "input":
            return [None, num_input]

    def _neural_net(self):
        # Define weights and biases for model
        # Set input tensor
        in_node = self.get_placeholder("input")
        self.set_placeholders([in_node])
        weights = dict(
            h1=tf.Variable(tf.random_normal([num_input, n_hidden_1], dtype=tf.float64),
                           name="l0W"),
            out=tf.Variable(tf.random_normal([n_hidden_1, num_classes], dtype=tf.float64),
                            name="l1W")
        )
        biases = dict(
            b1=tf.Variable(tf.random_normal([n_hidden_1], dtype=tf.float64), name="l0B"),
            out=tf.Variable(tf.random_normal([num_classes], dtype=tf.float64), name="l1B")
        )
        # Define model
        layer_1 = tf.nn.bias_add(tf.matmul(in_node, weights['h1']), biases['b1'], name="l0Preout")
        layer_1_post_actv = tf.abs(layer_1, name="l0Out")
        logits = tf.nn.bias_add(tf.matmul(layer_1_post_actv, weights['out']), biases['out'], name="l1PreOut")
        out_node = tf.nn.softmax(logits, name='output')
        self.set_output_tensors([out_node])


def test_vanilla_mlp():
    # Init TFP instance
    tfp = VanillaMLP(save_dir="mlp_00", seed=1337)
    # Run and persist
    tfp.build_and_save_graph()


if __name__ == '__main__':
    test_vanilla_mlp()
