import numpy as np

from tfoptests import load_save_utils

model_name = "non2d_0"
save_dir = model_name


def get_input(name):
    np.random.seed(19)
    if name == "scalar":
        input_1 = np.random.uniform()
        load_save_utils.save_input(input_1, name, save_dir)
        return input_1
    '''
    if name == "vector":
        input_2 = np.random.uniform(size=2)
        load_save_utils.save_input(input_2, name, save_dir)
        return input_2
    if name == "rank2dF":
        input_3 = np.random.uniform(size=(1, 2))
        load_save_utils.save_input(input_3, name, save_dir)
        return input_3
    if name == "rank2dB":
        input_3 = np.random.uniform(size=(2, 1))
        load_save_utils.save_input(input_3, name, save_dir)
        return input_3
    if name == "rank3d":
        input_4 = np.random.uniform(size=(1, 3, 2))
        load_save_utils.save_input(input_4, name, save_dir)
        return input_4
    if name == "rank3dA":
        input_4 = np.random.uniform(size=(3, 1, 2))
        load_save_utils.save_input(input_4, name, save_dir)
    if name == "rank4d":
        input_5 = np.random.uniform(size=(1, 4, 3, 2))
        load_save_utils.save_input(input_5, name, save_dir)
        return input_5
    '''


def list_inputs():
    return ["scalar"]


def get_inputs():
    my_input_dict = {}
    for a_input in list_inputs():
        my_input_dict[a_input] = get_input(a_input)
    return my_input_dict
