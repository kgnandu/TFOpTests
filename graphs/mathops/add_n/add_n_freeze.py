from graphs.mathops.add_n import save_dir, get_inputs
from helper import load_save_utils

load_save_utils.freeze_n_save_graph(save_dir)
load_save_utils.write_frozen_graph_txt(save_dir)
load_save_utils.save_intermediate_nodes(save_dir, get_inputs())
