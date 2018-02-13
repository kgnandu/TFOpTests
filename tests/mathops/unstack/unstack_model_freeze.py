from tests.mathops.unstack import save_dir
from tfoptests import load_save_utils
import numpy as np


load_save_utils.freeze_n_save_graph(save_dir)
load_save_utils.write_frozen_graph_txt(save_dir)
load_save_utils.save_intermediate_nodes(save_dir,{})