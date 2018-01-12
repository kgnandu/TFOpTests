from helper import load_save_utils
import numpy as np

model_name = "lstm_mnist"
save_dir = model_name

num_input = 28  # MNIST data input (img shape: 28*28)
timesteps = 28  # timesteps

def get_input(name, mnist):
    np.random.seed(13)
    if name == "input":
        test_len = 128
        input = mnist.test.images[:test_len].reshape((-1, timesteps, num_input))
        load_save_utils.save_input(input, "input", save_dir)
        return input


def list_inputs():
    return ["input"]


def get_inputs(mnist):
    my_input_dict = {}
    for a_input in list_inputs():
        my_input_dict[a_input] = get_input(a_input, mnist)
    return my_input_dict
