from tests.ae import  save_dir
from tfoptests import persistor
import numpy as np

n_in = 676

persistor.freeze_n_save_graph(save_dir)
persistor.write_frozen_graph_txt(save_dir)
input = np.reshape(np.linspace(1,n_in,n_in),(1,n_in))
persistor.save_intermediate_nodes(save_dir,{'input:0': input,'input':input})