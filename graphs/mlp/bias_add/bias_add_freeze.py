from graphs.mlp.bias_add import BiasAddInput, get_tf_persistor

persistor = get_tf_persistor()
inputs = BiasAddInput()

persistor.freeze_n_save_graph()
persistor.write_frozen_graph_txt()
persistor.save_intermediate_nodes(inputs())
