from abc import ABCMeta, abstractmethod
from itertools import izip
import tensorflow as tf

from helper import load_save_utils


class AbstractGraph(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.graph_placeholders = None
        self.graph_output_tensors = None
        '''
        Define graph in derived class and assign proper values to the above
        '''

    @abstractmethod
    def _get_input(self, name):
        pass

    @abstractmethod
    def _get_input_shape(self, name):
        pass

    def _get_input_dicts(self):
        placeholder_feed_dict = {}
        placeholder_name_value_dict = {}
        for input_tensor in self.graph_placeholders:
            input_name = input_tensor.name.split(":")[0]
            input_value = self._get_input(input_name)
            placeholder_feed_dict[input_tensor] = input_value
            placeholder_name_value_dict[input_name] = input_value
            load_save_utils.save_input(input_value, input_name, self.save_dir)
        return [placeholder_feed_dict, placeholder_name_value_dict]

    def get_placeholder(self, name, data_type="float64"):
        return tf.placeholder(dtype=data_type, shape=self._get_input_shape(name), name=name)

    def _check_outputs(self):
        for a_output in self.graph_output_tensors:
            if isinstance(a_output, list):
                raise ValueError('Output tensor elements cannot be lists...')

    def _list_output_node_names(self):
        output_node_names = []
        for a_output in self.graph_output_tensors:
            output_node_names.append(a_output.name.split(":")[0])
        return output_node_names

    def run_and_save_graph(self, please_print=True):
        placeholder_feed_dict, placeholder_name_value_dict = self._get_input_dicts()
        self._check_outputs()
        init = tf.global_variables_initializer()
        all_saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(init)
            predictions = sess.run(self.graph_output_tensors, feed_dict=placeholder_feed_dict)
            if please_print:
                print predictions
            load_save_utils.save_predictions(self.save_dir, dict(izip(self._list_output_node_names(), predictions)))
            load_save_utils.save_graph(sess, all_saver, self.save_dir)
        load_save_utils.freeze_n_save_graph(self.save_dir, output_node_names=",".join(self._list_output_node_names()))
        load_save_utils.write_frozen_graph_txt(self.save_dir)
        load_save_utils.save_intermediate_nodes(self.save_dir, placeholder_name_value_dict)
