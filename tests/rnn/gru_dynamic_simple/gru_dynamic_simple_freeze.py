from tests.rnn.gru_dynamic_simple import get_inputs, save_dir
from tfoptests import persistor

persistor.freeze_n_save_graph(save_dir)
persistor.write_frozen_graph_txt(save_dir)
persistor.save_intermediate_nodes(save_dir, get_inputs())