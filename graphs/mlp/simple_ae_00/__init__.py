import numpy as np

from helper import load_save_utils

model_name = "ae_00"
save_dir = model_name

my_input = np.array([[2.0, 1.0, 1.0, 2.0],
                     [-2.0, 1.0, -1.0, 2.0],
                     [0.0, 1.0, 0.0, 2.0],
                     [0.0, -1.0, 0.0, -2.0],
                     [0.0, -1.0, 0.0, -2.0]])

my_output = np.array([[2.0, 1.0, 1.0, 2.0],
                      [-2.0, 1.0, -1.0, 2.0],
                      [0.0, 1.0, 0.0, 2.0],
                      [0.0, -1.0, 0.0, -2.0],
                      [0.0, -1.0, 0.0, -2.0]])
np.random.seed(13)
noisy_input = my_input + .2 * np.random.random_sample((my_input.shape)) - .1

# Scale to [0,1]
scaled_input_1 = np.divide((noisy_input - noisy_input.min()), (noisy_input.max() - noisy_input.min()))
scaled_output_1 = np.divide((my_output - my_output.min()), (my_output.max() - my_output.min()))
# Scale to [-1,1]
scaled_input_2 = (scaled_input_1 * 2) - 1
scaled_output_2 = (scaled_output_1 * 2) - 1
output_data = scaled_output_2

def get_input(name):
    input_data = scaled_input_2
    if name == "input":
        load_save_utils.save_input(input_data, "input", save_dir)
        return input_data


def list_inputs():
    return ["input"]


def get_inputs():
    my_input_dict = {}
    for a_input in list_inputs():
        my_input_dict[a_input] = get_input(a_input)
    return my_input_dict
