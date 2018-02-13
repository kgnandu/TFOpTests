from tests.mlp.mlp_00 import BaseMLPInput, get_tf_persistor

persistor = get_tf_persistor()
inputs = BaseMLPInput()

persistor.freeze_n_save_graph()
persistor.write_frozen_graph_txt()
persistor.save_intermediate_nodes(inputs())
