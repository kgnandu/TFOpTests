import numpy as np

from helper.load_save_utils import TensorFlowPersistor

save_dir = "mnist_00"
PERSISTOR = TensorFlowPersistor(save_dir)


def get_tf_persistor():
    return PERSISTOR


class ModifiedMnistInput(TensorFlowPersistor):

    def get_input(self, name, mnist):
        np.random.seed(13)
        if name == "input":
            input = mnist.test.images[:100, :]
            PERSISTOR.save_input(input, "input")
            return input

    def __call__(self, mnist):
        my_input_dict = {}
        for a_input in self.list_inputs():
            my_input_dict[a_input] = self.get_input(a_input, mnist)
        return my_input_dict
