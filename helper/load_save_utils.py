from __future__ import print_function

import os
import tensorflow as tf
import numpy as np
from tensorflow.python.tools import freeze_graph

BASE_DIR = os.environ['DL4J_TEST_RESOURCES'] + '/src/main/resources/tf_graphs/examples'

# TODO: docstrings


class InputDictionary():

    def get_input(self, name):
        raise NotImplementedError

    def __call__(self):
        raise NotImplementedError

    def list_inputs(self):
        return ["input"]


class TensorFlowPersistor():

    def __init__(self, save_dir, base_dir=None):
        self.save_dir = save_dir
        self.base_dir = BASE_DIR if base_dir is None else base_dir

    def save_input(self, nparray, varname, name='placeholder'):
        contentfile = "{}/{}/{}.{}.csv".format(self.base_dir, self.save_dir, varname, name)
        shapefile = "{}/{}/{}.{}.shape".format(self.base_dir, self.save_dir, varname, name)

        if (np.isscalar(nparray)):
            np.savetxt(shapefile, np.asarray([0]), fmt="%i")
            with open(contentfile, 'w') as f:
                f.write('{}'.format(nparray))
        else:
            np.savetxt(shapefile, np.asarray(nparray.shape), fmt="%i")
            np.savetxt(contentfile, np.ndarray.flatten(nparray), fmt="%10.8f")

    def save_intermediate(self, nparray, varname, name='prediction_inbw'):
        self.safe_input(nparray, varname, name)

    def save_graph(self, sess, all_saver, data_path="data-all", model_file="model.txt"):
        all_saver.save(sess, "{}/{}/{}".format(self.base_dir, self.save_dir, data_path),
                       global_step=1000)
        tf.train.write_graph(sess.graph_def, "{}/{}".format(self.base_dir, self.save_dir),
                             model_file, True)

    def freeze_n_save_graph(self, output_node_names="output",
                            restore_op_name="save/restore_all",
                            filename_tensor_name="save/Const:0",
                            verbose=False):
        try:
            checkpoint = tf.train.get_checkpoint_state("{}/{}/".format(self.base_dir, self.save_dir))
            input_checkpoint = checkpoint.model_checkpoint_path
        except:
            raise ValueError("Could not read checkpoint state for path {}/{}"
                             .format(self.base_dir, self.save_dir))
        if verbose:
            print(input_checkpoint)
        output_graph = "{}/{}/frozen_model.pb".format(self.base_dir, self.save_dir)
        input_graph = "{}/{}/model.txt".format(self.base_dir, self.save_dir)
        freeze_graph.freeze_graph(input_graph=input_graph,
                                  input_saver="",
                                  input_checkpoint=input_checkpoint,
                                  output_graph=output_graph,
                                  input_binary=False,
                                  output_node_names=output_node_names,
                                  restore_op_name=restore_op_name,
                                  filename_tensor_name=filename_tensor_name,
                                  clear_devices=True,
                                  initializer_nodes="")

    def save_prediction(self, output):
        contentfile = BASE_DIR + "/" + self.save_dir + "/" + "output.prediction.csv"
        shapefile = BASE_DIR + "/" + self.save_dir + "/" + "output.prediction.shape"
        np.savetxt(shapefile, np.asarray(output.shape), fmt="%i")
        np.savetxt(contentfile, np.ndarray.flatten(output), fmt="%10.8f")

    def write_frozen_graph_txt(self, model_file='frozen_model.pb'):
        graph_filename = "{}/{}/{}".format(self.base_dir, self.save_dir, model_file)
        with tf.gfile.GFile(graph_filename, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.train.write_graph(graph_def, "{}/{}/".format(self.base_dir, self.save_dir),
                                 'frozen_graph.pbtxt', True)

    def load_frozen_graph(self, model_file='frozen_model.pb'):
        graph_filename = "{}/{}/{}".format(self.base_dir, self.save_dir, model_file)
        graph = tf.Graph()
        with tf.gfile.GFile(graph_filename, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        with graph.as_default():
            tf.import_graph_def(graph_def)
        return graph

    def save_intermediate_nodes(self, input_dict):
        graph = self.load_frozen_graph(self.save_dir)
        placeholder_dict = {}
        for op in graph.get_operations():
            if op.type != "Placeholder":
                continue
            print(op.name)  # there is a prefix and a suffix - there should only be one prefix
            placeholder_name = "/".join(op.name.split("/")[1:])
            placeholder_dict[op.name + ":0"] = input_dict[placeholder_name]

        for op in graph.get_operations():
            if op.type == "Placeholder":
                continue
            print("=====\n", op.name)
            output_num = 0
            for op_output in op.outputs:
                print(op_output.name)
                with tf.Session(graph=graph) as sess:
                    op_prediction = sess.run(op_output, feed_dict=placeholder_dict)
                    save_to = ".".join(["____".join(op_output.name.split("/")[1:]).split(":")[0], str(output_num)])
                    self.save_intermediate(op_prediction, save_to, self.save_dir)
                    print(op_prediction, "\n=====")
                output_num += 1
