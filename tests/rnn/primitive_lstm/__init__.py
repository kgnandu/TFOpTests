import numpy as np
from tfoptests.load_save_utils import TensorFlowPersistor

save_dir = "primitive_lstm"
PERSISTOR = TensorFlowPersistor(save_dir)

'''
Sequence is just add 0.01
input shape is ((-1, timesteps, num_input))
'''
timesteps = 7
featuresize = 2
minibatch = 5

np.random.seed(13)
l = np.linspace(0, timesteps - 1, timesteps).reshape(1, timesteps) / 100
timeseq = np.zeros((minibatch, featuresize, timesteps))
timeseq += l
r = np.random.uniform(0, 0.5, minibatch * featuresize).reshape(minibatch, featuresize, 1)
timeseq += r


def get_tf_persistor():
    return PERSISTOR


class SimpleLstmInput(TensorFlowPersistor):

    def __init__(self):
        self.featuresize = featuresize
        self.timesteps = timesteps
        self.minibatch = minibatch

    def get_input(self, name):
        np.random.seed(13)
        if name == "input":
            return timeseq[:, :, :-1]

    def get_output(self, name):
        if name == "output":
            return timeseq[:, :, -1].reshape((minibatch, featuresize))

    def get_inputs(self):
        my_input_dict = {}
        for a_input in self.list_inputs():
            my_input_dict[a_input] = self.get_input(a_input)
        return my_input_dict
