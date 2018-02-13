import numpy as np

from tfoptests import persistor

model_name = "g_05"
save_dir = model_name


def get_input(name):
    np.random.seed(19)
    if name == "input_1":
        input_1 = np.random.uniform(size=(2,4,3,2))
        persistor.save_input(input_1, name, save_dir)
        return input_1
    if name == "input_2":
        input_2 = np.random.uniform(size=(3, 2))
        persistor.save_input(input_2, name, save_dir)
        return input_2


def list_inputs():
    return ["input_1", "input_2"]


def get_inputs():
    my_input_dict = {}
    for a_input in list_inputs():
        my_input_dict[a_input] = get_input(a_input)
    return my_input_dict