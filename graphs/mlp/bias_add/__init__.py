from helper.load_save_utils import TensorFlowPersistor, InputDictionary
import numpy as np

np.random.seed(13)
save_dir = "bias_add"
PERSISTOR = TensorFlowPersistor(save_dir)


def get_tf_persistor():
    return PERSISTOR


class BiasAddInput(InputDictionary):

    def get_input(self, name):
        if name == "input":
            input_0 = np.linspace(1, 40, 40).reshape(10, 4)
            PERSISTOR.save_input(input_0, "input")
            return input_0

    def __call__(self):
        my_input_dict = {}
        for a_input in self.list_inputs():
            my_input_dict[a_input] = self.get_input(a_input)
        return my_input_dict
