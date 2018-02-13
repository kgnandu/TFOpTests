import numpy as np

from tfoptests.load_save_utils import TensorFlowPersistor

save_dir = "mlp_00"
n_hidden_1 = 10  # number of neurons in first layer
num_input = 5
num_classes = 3
mini_batch = 4

PERSISTOR = TensorFlowPersistor(save_dir)


def get_tf_persistor():
    return PERSISTOR


class BaseMLPInput(TensorFlowPersistor):

    def __init__(self):
        self.n_hidden_1 = n_hidden_1
        self.num_input = num_input
        self.num_classes = num_classes

    def get_input(self, name):
        np.random.seed(13)
        if name == "input":
            input_0 = np.random.uniform(size=(mini_batch, num_input))
            PERSISTOR.save_input(input_0, "input")
            return input_0

    def __call__(self):
        my_input_dict = {}
        for a_input in self.list_inputs():
            my_input_dict[a_input] = self.get_input(a_input)
        return my_input_dict
