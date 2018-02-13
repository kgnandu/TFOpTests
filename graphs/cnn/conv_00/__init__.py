from helper.load_save_utils import TensorFlowPersistor
import numpy as np

save_dir = "conv_0"
PERSISTOR = TensorFlowPersistor(save_dir)

imsize = [4, 28, 28, 3]


def get_tf_persistor():
    return PERSISTOR


class BaseCNNInput(TensorFlowPersistor):

    def __init__(self):
        self.imsize = imsize

    def get_input(self, name):
        np.random.seed(13)
        if name == "input_0":
            input_0 = np.random.uniform(size=imsize)
            PERSISTOR.save_input(input_0, name, save_dir)
            return input_0

    def get_inputs(self):
        my_input_dict = {}
        for a_input in self.list_inputs():
            my_input_dict[a_input] = self.get_input(a_input)
        return my_input_dict
