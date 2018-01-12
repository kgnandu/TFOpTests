import numpy as np

from helper import load_save_utils

model_name = "non2d_0A"
save_dir = model_name


def get_input(name):
    np.random.seed(19)
    if name == "scalarA":
        input_1 = np.random.random_integers(0,6)
        load_save_utils.save_input(input_1, name, save_dir)
        return input_1
    if name == "scalarB":
        input_1 = np.random.random_integers(0,4)
        load_save_utils.save_input(input_1, name, save_dir)
        return input_1
    if name == "some_weight":
        input_2 = np.random.uniform(size=(3,4))
        return input_2


def list_inputs():
    return ["scalarA", "scalarB"]


def get_inputs():
    my_input_dict = {}
    for a_input in list_inputs():
        my_input_dict[a_input] = get_input(a_input)
    return my_input_dict
