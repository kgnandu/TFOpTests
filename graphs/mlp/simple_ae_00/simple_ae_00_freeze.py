from graphs.mlp.simple_ae_00 import AutoEncoderInput, get_tf_persistor

persistor = get_tf_persistor()
inputs = AutoEncoderInput()

persistor.freeze_n_save_graph()
persistor.write_frozen_graph_txt()
persistor.save_intermediate_nodes(inputs())
