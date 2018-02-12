from helper.load_save_utils import TensorFlowPersistor, InputDictionary
import numpy as np

save_dir = "lstm_mnist"
PERSISTOR = TensorFlowPersistor(save_dir)

num_input = 28  # MNIST data input (img shape: 28*28)
timesteps = 28  # timesteps


def get_tf_persistor():
    return PERSISTOR


class MnistLstmInput(InputDictionary):

    def __init__(self):
        self.num_input = num_input
        self.timesteps = timesteps

    def get_input(self, name, mnist):
        np.random.seed(13)
        if name == "input":
            test_len = 128
            input = mnist.test.images[:test_len].reshape((-1, timesteps, num_input))
            PERSISTOR.save_input(input, "input", save_dir)
            return input

    def get_inputs(self, mnist):
        my_input_dict = {}
        for a_input in self.list_inputs():
            my_input_dict[a_input] = self.get_input(a_input, mnist)
        return my_input_dict
