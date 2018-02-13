import numpy as np
import tensorflow as tf
from tfoptests.load_save_utils import TensorFlowPersistor

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
            return [mini_batch, num_input]


def neural_net(input):
    # Define weights and biases for model
    weights = dict(
        h1=tf.Variable(tf.random_normal([num_input, n_hidden_1]),
                       name="l0W", dtype=tf.float32),
        out=tf.Variable(tf.random_normal([n_hidden_1, num_classes]),
                        name="l1W", dtype=tf.float32)
    )
    biases = dict(
        b1=tf.Variable(tf.random_normal([n_hidden_1]), name="l0B"),
        out=tf.Variable(tf.random_normal([num_classes]), name="l1B")
    )
    # Define model
    layer_1 = tf.nn.bias_add(tf.matmul(input, weights['h1']), biases['b1'], name="l0Preout")
    layer_1_post_actv = tf.abs(layer_1, name="l0Out")
    out_layer = tf.nn.bias_add(tf.matmul(layer_1_post_actv, weights['out']), biases['out'], name="l1PreOut")
    return out_layer


if __name__ == '__main__':
    # Init TFP instance
    tfp = VanillaMLP(save_dir="mlp_00", seed=1337)

    # Set input tensor
    in_node = tfp.get_placeholder("input")
    tfp.set_placeholders([in_node])

    #  Construct model and set output tensor
    input = tf.placeholder("float", tfp._get_input_shape("input"), name="input")
    logits = neural_net(input)
    out_node = tf.nn.softmax(logits, name='output')
    tfp.set_output_tensors([out_node])

    # Run and persist
    tfp.run_and_save_graph()
