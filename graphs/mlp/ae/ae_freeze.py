from graphs.mlp.ae import save_dir
from helper import load_save_utils
import numpy as np

n_in = 676

load_save_utils.freeze_n_save_graph(save_dir)
load_save_utils.write_frozen_graph_txt(save_dir)
input = np.reshape(np.linspace(1, n_in, n_in), (1, n_in))
load_save_utils.save_intermediate_nodes(save_dir, {'input:0': input, 'input': input})
