from tfoptests.persistor import TensorFlowPersistor
from tfoptests.test_graph import TestGraph
import tensorflow as tf
import numpy as np


class TwoInputs(TestGraph):
    def __init__(self, *args, **kwargs):
        super(TwoInputs, self).__init__(*args, **kwargs)
        self.input_0 = np.linspace(1, 4, 4).reshape(2, 2)
        self.input_1 = 11

    def list_inputs(self):
        return ["input_0", "input_1"]

    def get_placeholder_input(self, name):
        if name == "input_0":
            return self.input_0
        if name == "input_1":
            return self.input_1

    def _get_placeholder_shape(self, name):
        if name == "input_0":
            return [2, 2]
        if name == "input_1":
            return []


def test_simplewhile_0():
    two_inputs = TwoInputs(seed=13)
    in0 = two_inputs.get_placeholder("input_0", data_type=tf.float32)
    in1 = two_inputs.get_placeholder("input_1", data_type=tf.float32)
    in3 = tf.Variable(tf.constant(2.0, shape=[], name="addVal"))

    def body(x):
        return tf.add(x, in3)

    def condition(x):
        return tf.less(tf.reduce_sum(x), in1)

    in0p = tf.while_loop(condition, body, [in0])
    out_node = tf.identity(in0p, name="output")
    placeholders = [in0, in1]
    predictions = [out_node]

    # Run and persist
    tfp = TensorFlowPersistor(save_dir="simplewhile_0")
    tfp.set_placeholders(placeholders) \
        .set_output_tensors(predictions) \
        .set_test_data(two_inputs.get_test_data()) \
        .build_save_frozen_graph()


if __name__ == '__main__':
    test_simplewhile_0()
