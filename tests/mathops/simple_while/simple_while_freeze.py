from tests.mathops.simple_while import save_dir, get_inputs
from tfoptests import persistor

persistor.freeze_n_save_graph(save_dir,output_node_names='output/Exit')
persistor.write_frozen_graph_txt(save_dir)
persistor.save_intermediate_nodes(save_dir,get_inputs())