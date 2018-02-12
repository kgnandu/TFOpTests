from tensorflow.examples.tutorials.mnist import input_data
import argparse
import sys
import tensorflow as tf

from graphs.cnn.deep_mnist_modified import DeepMnistModified, get_tf_persistor

persistor = get_tf_persistor()
inputs = DeepMnistModified()

FLAGS = None


def main(_):
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
    persistor.freeze_n_save_graph()
    persistor.write_frozen_graph_txt()
    persistor.save_intermediate_nodes(inputs(mnist))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        default='/tmp/tensorflow/mnist/input_data',
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
