from tfoptests.load_save_utils import TensorFlowPersistor
import numpy as np

save_dir = "deep_mnist_no_dropout"
PERSISTOR = TensorFlowPersistor(save_dir)


def get_tf_persistor():
    return PERSISTOR


class DeepMnistModified(TensorFlowPersistor):

    def get_input(self, name, mnist):
        np.random.seed(13)
        if name == "input":
            input = mnist.test.images[:100, :]
            PERSISTOR.save_input(input, "input", save_dir)
            return input

    def get_inputs(self, mnist):
        my_input_dict = {}
        for a_input in self.list_inputs():
            my_input_dict[a_input] = self.get_input(a_input, mnist)
        return my_input_dict
