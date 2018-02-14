import numpy as np

from helper import load_save_utils

model_name = "simple_cond"
save_dir = model_name

def get_input(name):
    if name == "greater":
        input_0 = np.reshape(np.linspace(1,4,4),(2,2))
        load_save_utils.save_input(input_0, name, save_dir)
        return input_0
    if name == "lesser":
        input_1 = np.reshape(np.linspace(1,4,4),(2,2))
        load_save_utils.save_input(input_1, name, save_dir)
        return input_1


def list_inputs():
    return []


def get_inputs():
    my_input_dict = {}
    for a_input in list_inputs():
        my_input_dict[a_input] = get_input(a_input)
    return my_input_dict