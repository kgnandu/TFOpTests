import numpy as np
from helper import load_save_utils

model_name = "g_04"
save_dir = model_name


def get_input(name):
    np.random.seed(19)
    if name == "input_1":
        input_1 = np.random.uniform(size=(16, 16))
        load_save_utils.save_input(input_1, name, save_dir)
        return input_1
    if name == "input_2":
        input_2 = np.random.uniform(size=(16, 16)) + np.random.uniform(size=(16, 16))
        load_save_utils.save_input(input_2, name, save_dir)
        return input_2


def list_inputs():
    return ["input_1", "input_2"]


def get_inputs():
    my_input_dict = {}
    for a_input in list_inputs():
        my_input_dict[a_input] = get_input(a_input)
    return my_input_dict