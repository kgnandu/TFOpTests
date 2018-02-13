import argparse
import sys

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

from tests.rnn.gru_dynamic_mnist import save_dir, get_inputs
from tfoptests import load_save_utils


def main(_):
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
    load_save_utils.freeze_n_save_graph(save_dir)
    load_save_utils.write_frozen_graph_txt(save_dir)
    # load_save_utils.save_intermediate_nodes(save_dir, get_inputs(mnist))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        default='/tmp/tensorflow/mnist/input_data',
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
