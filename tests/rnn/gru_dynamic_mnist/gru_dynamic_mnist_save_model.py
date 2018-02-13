# Example for my blog post at:
# https://danijar.com/introduction-to-recurrent-networks-in-tensorflow/
import functools
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

from tests.rnn.gru_dynamic_mnist import num_input, timesteps, save_dir, get_input
from tfoptests import persistor

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
num_classes = 10
# tf Graph input
X = tf.placeholder("float", [None, timesteps, num_input], name="input")
Y = tf.placeholder("float", [None, num_classes])


def lazy_property(function):
    attribute = '_' + function.__name__

    @property
    @functools.wraps(function)
    def wrapper(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return wrapper


class SequenceClassification:
    def __init__(self, data, target, num_hidden=200, num_layers=3):
        self.data = data
        self.target = target
        self._num_hidden = num_hidden
        self._num_layers = num_layers
        self.prediction
        self.error
        self.optimize

    @lazy_property
    def prediction(self):
        # Recurrent network.
        cells = []
        for _ in range(self._num_layers):
            cell = tf.contrib.rnn.GRUCell(self._num_hidden)  # Or LSTMCell(num_units)
            cells.append(cell)
        network = tf.contrib.rnn.MultiRNNCell(cells)
        output, _ = tf.nn.dynamic_rnn(network, self.data, dtype=tf.float32)
        # Select last output.
        output = tf.transpose(output, [1, 0, 2])
        last = tf.gather(output, int(output.get_shape()[0]) - 1)
        # Softmax layer.
        weight, bias = self._weight_and_bias(
            self._num_hidden, int(self.target.get_shape()[1]))
        prediction = tf.nn.softmax(tf.matmul(last, weight) + bias, name="output")
        return prediction

    @lazy_property
    def cost(self):
        cross_entropy = -tf.reduce_sum(self.target * tf.log(self.prediction))
        return cross_entropy

    @lazy_property
    def optimize(self):
        learning_rate = 0.003
        optimizer = tf.train.RMSPropOptimizer(learning_rate)
        return optimizer.minimize(self.cost)

    @lazy_property
    def error(self):
        mistakes = tf.not_equal(
            tf.argmax(self.target, 1), tf.argmax(self.prediction, 1))
        return tf.reduce_mean(tf.cast(mistakes, tf.float32))

    @staticmethod
    def _weight_and_bias(in_size, out_size):
        weight = tf.truncated_normal([in_size, out_size], stddev=0.01)
        bias = tf.constant(0.1, shape=[out_size])
        return tf.Variable(weight), tf.Variable(bias)


def main():
    # We treat images as sequences of pixel rows.
    # train,test = mnist.train,mnist.test
    model = SequenceClassification(X, Y)
    sess = tf.Session()
    all_saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    test_len = 128
    test_data = mnist.test.images[:test_len].reshape((-1, timesteps, num_input))
    test_target = mnist.test.labels[:test_len]
    for epoch in range(10):
        for _ in range(100):
            # batch = train.sample(10)
            batch_data, batch_target = mnist.train.next_batch(10)
            sess.run(model.optimize, {
                X: batch_data.reshape((-1, 28, 28)), Y: batch_target})
        error = sess.run(model.error, {
            X: test_data, Y: test_target})
        print('Epoch {:2d} error {:3.1f}%'.format(epoch + 1, 100 * error))
    prediction = sess.run(model.prediction, feed_dict={X: get_input("input", mnist)})
    persistor.save_prediction(save_dir, prediction)
    persistor.save_graph(sess, all_saver, save_dir)


if __name__ == '__main__':
    main()
