import numpy as np
from tfoptests import persistor

model_name = "conv_2"
save_dir = model_name
# [batch, in_height, in_width, in_channels].
imsize = [4, 64, 64, 4]

def get_input(name):
    np.random.seed(13)
    if name == "input_0":
        input_0 = np.random.uniform(size=imsize)
        persistor.save_input(input_0, name, save_dir)
        return input_0


def list_inputs():
    return ["input_0"]


def get_inputs():
    my_input_dict = {}
    for a_input in list_inputs():
        my_input_dict[a_input] = get_input(a_input)
    return my_input_dict