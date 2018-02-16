from __future__ import print_function

from abc import ABCMeta, abstractmethod

try:
    from itertools import izip as zip
except:
    # just use plain zip in py3
    pass
import sys
import os
import tensorflow as tf
import numpy as np
from tensorflow.python.tools import freeze_graph

BASE_DIR = os.environ['DL4J_TEST_RESOURCES'] + '/src/main/resources/tf_graphs/examples'


# BASE_DIR = '/Users/susaneraly/SKYMIND/dl4j-test-resources/src/main/resources/tf_graphs/examples'


class TensorFlowPersistor():
    '''
    TensorFlowPersistor (TFP) is the main abstraction of this module. A TFP
    has all the functionality to load and store tensorflow tests.

    TFP is an abstract base class. You need to implement `get_input`
    and `get_input_shape` for the graph data of your choice.
    '''
    __metaclass__ = ABCMeta

    def __init__(self, save_dir, seed=None, base_dir=None, verbose=True):
        self.save_dir = save_dir
        self.base_dir = BASE_DIR if base_dir is None else base_dir
        self.seed = None
        self.verbose = verbose
        self._graph_placeholders = None
        self._graph_output_tensors = None

    @abstractmethod
    def _get_input(self, name):
        '''Get input tensor for given node name'''
        raise NotImplementedError

    @abstractmethod
    def _get_input_shape(self, name):
        '''Get input tensor shape for given node name'''
        raise NotImplementedError

    @abstractmethod
    def _neural_net(self):
        '''Implement the neural net and if training return the TF sess. TODO: Document better with example'''
        raise NotImplementedError

    @property
    def graph_placeholders(self):
        '''
        graph_placeholders: Input variables to the tensorflow graph

        Example:
        in_node = tfp.get_placeholder("input")
        graph_placeholders = [in_node]
        '''
        return self._graph_placeholders

    def set_placeholders(self, graph_placeholders):
        self._graph_placeholders = graph_placeholders

    @property
    def graph_output_tensors(self):
        '''
        graph_placeholders: Input variables to the tensorflow graph
        graph_output_tensors: Output variables of the tensorlow graph

        Example:
        in_node = tfp.get_placeholder("input")
        biases = tf.Variable(tf.lin_space(1.0, 4.0, 4), name="bias")
        out_node = tf.nn.bias_add(in_node, tf.cast(biases, dtype=tf.float64), name="output")
        graph_output_tensors = [out_node]
        '''
        return self._graph_output_tensors

    def set_output_tensors(self, graph_output_tensors):
        self._graph_output_tensors = graph_output_tensors

    def _write_to_file(self, nparray, content_file, shape_file):
        if np.isscalar(nparray):
            np.savetxt(shape_file, np.asarray([0]), fmt="%i")
            f = open(content_file, 'w')
            f.write('{}'.format(nparray))
            f.close()
        else:
            np.savetxt(shape_file, np.asarray(nparray.shape), fmt="%i")
            np.savetxt(content_file, np.ndarray.flatten(nparray), fmt="%10.8f")

    def _save_content(self, nparray, varname, name):
        content_file = "{}/{}/{}.{}.csv".format(self.base_dir, self.save_dir, varname, name)
        shape_file = "{}/{}/{}.{}.shape".format(self.base_dir, self.save_dir, varname, name)
        self._write_to_file(nparray, content_file, shape_file)

    def save_input(self, nparray, varname, name='placeholder'):
        self._save_content(nparray, varname, name)

    def save_intermediate(self, nparray, varname, name='prediction_inbw'):
        self._save_content(nparray, varname, name)

    def save_prediction(self, output, varname='output', name='prediction'):
        self._save_content(output, varname, name)

    def save_predictions(self, output_dict, name='prediction'):
        for output_name, output_value in output_dict.items():
            self._save_content(output_value, output_name, name)

    def save_graph(self, sess, all_saver, data_path="data-all", model_file="model.txt"):
        all_saver.save(sess, "{}/{}/{}".format(self.base_dir, self.save_dir, data_path),
                       global_step=1000)
        tf.train.write_graph(sess.graph_def, "{}/{}".format(self.base_dir, self.save_dir),
                             model_file, True)

    def freeze_n_save_graph(self, output_node_names="output",
                            restore_op_name="save/restore_all",
                            filename_tensor_name="save/Const:0"):
        try:
            checkpoint = tf.train.get_checkpoint_state("{}/{}/".format(self.base_dir, self.save_dir))
            input_checkpoint = checkpoint.model_checkpoint_path
        except:
            raise ValueError("Could not read checkpoint state for path {}/{}"
                             .format(self.base_dir, self.save_dir))
        if self.verbose:
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
        graph = self.load_frozen_graph()
        placeholder_dict = {}
        prediction_dict = {}
        if self.verbose:
            print("-----------------------------------------------------")
            print("PLACEHOLDER LIST:")
        for op in graph.get_operations():
            if op.type != "Placeholder":
                continue
            if self.verbose:
                print(op.name)  # there is a prefix and a suffix - there should only be one prefix
            placeholder_name = "/".join(op.name.split("/")[1:])
            placeholder_dict[op.name + ":0"] = input_dict[placeholder_name]
        if self.verbose:
            print("-----------------------------------------------------")

        for op in graph.get_operations():
            if op.type == "Placeholder":
                continue
            if self.verbose:
                print(op.name)
                print(op.type)
            for op_output in op.outputs:
                if self.verbose:
                    print(op_output.name)
                with tf.Session(graph=graph) as sess:
                    try:
                        if op_output.dtype.is_bool:
                            if self.verbose:
                                print("SKIPPING bool")
                                print("-----------------------------------------------------")
                        else:
                            op_prediction = sess.run(op_output, feed_dict=placeholder_dict)
                            tensor_output_name = ("/".join(op_output.name.split("/")[1:])).split(":")[0]
                            tensor_output_num = op_output.name.split(":")[1]
                            if tensor_output_name in self._list_output_node_names():
                                prediction_dict[tensor_output_name] = op_prediction
                            if self.verbose:
                                print(op_prediction)
                                print("-----------------------------------------------------")
                            modified_tensor_output_name = "____".join(tensor_output_name.split("/"))
                            save_to = ".".join([modified_tensor_output_name, tensor_output_num])
                            self.save_intermediate(op_prediction, save_to)
                    except:
                        print("Unexpected error:", sys.exc_info()[0])
                        if self.verbose:
                            print(op_output)
                            print("SKIPPING")
                            print("-----------------------------------------------------")
        return prediction_dict

    def load_external_graph(self, model_file):
        graph = tf.Graph()
        graph_def = tf.GraphDef()

        with open(model_file, "rb") as f:
            graph_def.ParseFromString(f.read())
        with graph.as_default():
            tf.import_graph_def(graph_def)

        return graph

    def _get_input_dicts(self):
        placeholder_feed_dict = {}
        placeholder_name_value_dict = {}
        for input_tensor in self.graph_placeholders:
            input_name = input_tensor.name.split(":")[0]
            input_value = self._get_input(input_name)
            placeholder_feed_dict[input_tensor] = input_value
            placeholder_name_value_dict[input_name] = input_value
            self.save_input(input_value, input_name)
        return [placeholder_feed_dict, placeholder_name_value_dict]

    def get_placeholder(self, name, data_type="float64"):
        return tf.placeholder(dtype=data_type, shape=self._get_input_shape(name), name=name)

    def _check_outputs(self):
        if self._graph_output_tensors is None:
            raise ValueError("Ouput tensor list not set")
        for a_output in self._graph_output_tensors:
            if isinstance(a_output, list):
                raise ValueError('Output tensor elements cannot be lists...')

    def _check_inputs(self):
        if self._graph_placeholders is None:
            raise ValueError("Input tensor placeholder list not set")

    def _list_output_node_names(self):
        output_node_names = []
        for a_output in self.graph_output_tensors:
            output_node_names.append(a_output.name.split(":")[0])
        return output_node_names

    def build_and_save_graph(self):
        sess, out_after_train = self._neural_net()  # build the neural net
        self._check_inputs()  # make sure input placeholders are set
        self._check_outputs()  # make sure outputs are set
        placeholder_feed_dict, placeholder_name_value_dict = self._get_input_dicts()
        all_saver = tf.train.Saver()
        if sess is None:
            init = tf.global_variables_initializer()
            with tf.Session() as sess:
                sess.run(init)
                predictions = sess.run(self.graph_output_tensors, feed_dict=placeholder_feed_dict)
        else:
            predictions = sess.run(self.graph_output_tensors, feed_dict=placeholder_feed_dict)
        first_pass_dict = dict(zip(self._list_output_node_names(), predictions))
        if self.verbose:
            print(predictions)
        self.save_graph(sess, all_saver)
        sess.close()
        self.save_predictions(first_pass_dict)
        self.freeze_n_save_graph(output_node_names=",".join(self._list_output_node_names()))
        self.write_frozen_graph_txt()
        second_pass_dict = self.save_intermediate_nodes(placeholder_name_value_dict)
        # Better way to do this assert??
        assert second_pass_dict.keys() == first_pass_dict.keys()
        for a_output in second_pass_dict.keys():
            np.testing.assert_equal(first_pass_dict[a_output], out_after_train[a_output])
            np.testing.assert_equal(first_pass_dict[a_output], second_pass_dict[a_output])
