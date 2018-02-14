from helper import load_save_utils
import numpy as np

model_name = "deep_mnist"
save_dir = model_name


def get_input(name, mnist):
    np.random.seed(13)
    if name == "input":
        input = mnist.test.images[:100, :]
        load_save_utils.save_input(input, "input", save_dir)
        return input
    if name == "keep_prob":
        input = 0.6
        load_save_utils.save_input(input, "keep_prob", save_dir)
        return input


def list_inputs():
    return ["input", "keep_prob"]


def get_inputs(mnist):
    my_input_dict = {}
    for a_input in list_inputs():
        my_input_dict[a_input] = get_input(a_input, mnist)
    return my_input_dict
