from helper import load_save_utils
import numpy as np

model_name = "node_multiple_out"
save_dir = model_name


def get_input(name):
    np.random.seed(13)
    if name == "input_0":
        input_0 = np.linspace(1, 120, 120).reshape(2, 3, 4, 5)
        load_save_utils.save_input(input_0, name, save_dir)
        return input_0


def list_inputs():
    return ["input_0"]


def get_inputs():
    my_input_dict = {}
    for a_input in list_inputs():
        my_input_dict[a_input] = get_input(a_input)
    return my_input_dict
