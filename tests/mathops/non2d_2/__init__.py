from tfoptests import persistor
import numpy as np

model_name = "non2d_2"
save_dir = model_name


def get_input(name):
    np.random.seed(19)
    if name == "rank2dF":
        input_3 = np.random.uniform(size=(1, 2))
        persistor.save_input(input_3, name, save_dir)
        return input_3
    if name == "rank2dB":
        input_3 = np.random.uniform(size=(2, 1))
        persistor.save_input(input_3, name, save_dir)
        return input_3
    if name == "rank3d":
        input_4 = np.random.uniform(size=(1, 3, 2))
        persistor.save_input(input_4, name, save_dir)
        return input_4
    if name == "rank3dB":
        input_4 = np.random.uniform(size=(3, 1, 2))
        persistor.save_input(input_4, name, save_dir)
        return input_4


def list_inputs():
    return ["rank2dF","rank2dB","rank3d","rank3dB"]


def get_inputs():
    my_input_dict = {}
    for a_input in list_inputs():
        my_input_dict[a_input] = get_input(a_input)
    return my_input_dict