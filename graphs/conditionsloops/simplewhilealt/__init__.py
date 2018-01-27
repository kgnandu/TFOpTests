from helper import load_save_utils
import numpy as np

model_name = "simplewhile_0_alt"
save_dir = model_name


def get_input(name):
    if name == "input_0":
        input_0 = np.linspace(1, 4, 4).reshape(2, 2)
        load_save_utils.save_input(input_0, name, save_dir)
        return input_0
    if name == "input_1":
        input_1 = 11
        load_save_utils.save_input(input_1, name, save_dir)
        return input_1


def list_inputs():
    return ["input_0", "input_1"]


def get_inputs():
    my_input_dict = {}
    for a_input in list_inputs():
        my_input_dict[a_input] = get_input(a_input)
    return my_input_dict