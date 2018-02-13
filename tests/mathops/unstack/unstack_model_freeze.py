from tests.mathops.unstack import save_dir
from tfoptests import persistor
import numpy as np


persistor.freeze_n_save_graph(save_dir)
persistor.write_frozen_graph_txt(save_dir)
persistor.save_intermediate_nodes(save_dir,{})