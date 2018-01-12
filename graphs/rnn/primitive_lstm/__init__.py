import numpy as np

from helper import load_save_utils

model_name = "primitive_lstm"
save_dir = model_name

'''
Sequence is just add 0.01 
input shape is ((-1, timesteps, num_input))
'''
timesteps = 7
featuresize = 2
minibatch = 5

np.random.seed(13)
l = np.linspace(0, timesteps - 1, timesteps).reshape(1, timesteps) / 100;
timeseq = np.zeros((minibatch, featuresize, timesteps))
timeseq += l
r = np.random.uniform(0, 0.5, minibatch * featuresize).reshape(minibatch, featuresize, 1)
timeseq += r


def get_input(name):
    np.random.seed(13)
    if name == "input":
        input = timeseq[:, :, :-1]
        load_save_utils.save_input(input, "input", save_dir)
        return input


def get_output(name):
    if name == "output":
        return timeseq[:, :, -1].reshape((minibatch, featuresize))


def list_inputs():
    return ["input"]


def get_inputs():
    my_input_dict = {}
    for a_input in list_inputs():
        my_input_dict[a_input] = get_input(a_input)
    return my_input_dict
