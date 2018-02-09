from abc import ABCMeta, abstractmethod
from itertools import izip
import tensorflow as tf

from helper import load_save_utils


class AbstractGraph(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, save_dir):
        self.save_dir = save_dir
        self._tensor_feed_dict = None
        self._input_names_with_value = None
        self.graph_placeholders = None
        self.graph_output_tensors = None
        '''
        Define graph in derived class and assign proper values to the above
        '''

    @abstractmethod
    def list_inputs(self):
        pass

    @abstractmethod
    def list_outputs(self):
        pass

    @abstractmethod
    def _get_input(self, name):
        pass

    @abstractmethod
    def _get_input_shape(self, name):
        pass

    def _load_input_dicts(self):
        self._tensor_feed_dict = {}
        self._input_names_with_value = {}
        for input_tensor in self.graph_placeholders:
            input_name = input_tensor.name.split(":")[0]
            input_value = self._get_input(input_name)
            self._tensor_feed_dict[input_tensor] = input_value
            self._input_names_with_value[input_name] = input_value
            load_save_utils.save_input(input_value, input_name, self.save_dir)

    def get_placeholder(self, name, data_type="float64"):
        return tf.placeholder(dtype=data_type, shape=self._get_input_shape(name), name=name)

    def _check_outputs(self):
        for a_output in self.graph_output_tensors:
            if isinstance(a_output, list):
                raise ValueError('Output tensor elements cannot be lists...')

    def run_and_save_graph(self, please_print=True):
        self._load_input_dicts()
        self._check_outputs()
        init = tf.global_variables_initializer()
        all_saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(init)
            predictions = sess.run(self.graph_output_tensors, feed_dict=self._tensor_feed_dict)
            if please_print:
                print predictions
            load_save_utils.save_predictions(self.save_dir, dict(izip(self.list_outputs(), predictions)))
            load_save_utils.save_graph(sess, all_saver, self.save_dir)
        load_save_utils.freeze_n_save_graph(self.save_dir, output_node_names=",".join(self.list_outputs()))
        load_save_utils.write_frozen_graph_txt(self.save_dir)
        load_save_utils.save_intermediate_nodes(self.save_dir, self._input_names_with_value)
