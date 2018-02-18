# from abc import ABCMeta, abstractmethod
import tensorflow as tf


class TestGraph:
    # __metaclass__ = ABCMeta

    def __init__(self, seed=None, verbose=True):
        self.verbose = verbose
        self.seed = seed

    # @abstractmethod
    def _get_placeholder_input(self, name):
        '''Get input tensor for given node name'''
        # raise NotImplementedError
        return None

    ## @abstractmethod
    def _get_placeholder_shape(self, name):
        '''Get input tensor shape for given node name'''
        # raise NotImplementedError
        return None

    ## @abstractmethod
    def list_inputs(self):
        '''List names of input nodes'''
        # raise NotImplementedError
        return [None]

    def get_placeholder(self, name, data_type="float64"):
        return tf.placeholder(dtype=data_type, shape=self._get_placeholder_shape(name), name=name)

    def get_test_data(self):
        test_dict = {}
        for an_input in self.list_inputs():
            test_dict[an_input] = self._get_placeholder_input(an_input)
        return test_dict
