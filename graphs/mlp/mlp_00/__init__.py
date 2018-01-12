import numpy as np

from helper import load_save_utils

model_name = "mlp_00"
save_dir = model_name
n_hidden_1 = 10  # 1st layer number of neurons
num_input = 5
num_classes = 3
mini_batch = 4


def get_input(name):
    np.random.seed(13)
    if name == "input":
        input_0 = np.random.uniform(size=(mini_batch, num_input))
        load_save_utils.save_input(input_0, "input", save_dir)
        return input_0


def list_inputs():
    return ["input"]


def get_inputs():
    my_input_dict = {}
    for a_input in list_inputs():
        my_input_dict[a_input] = get_input(a_input)
    return my_input_dict
