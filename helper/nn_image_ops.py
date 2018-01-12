import tensorflow as tf
import numpy as np

'''
http://mourafiq.com/2016/08/10/playing-with-convolutions-in-tensorflow.html
'''


class NNImageOps:
    def __init__(self, img):
        self.inp = img
        self.node_num = 0
        self.kernel_size = None
        self.strides = None
        self.padding = 'SAME'
        self.filter_size = None
        self.filter = None

    def set_image(self, image):
        # [batch, in_height, in_width, in_channels]
        self.inp = image

    def set_kernel(self, some_l):
        self.kernel_size = some_l

    def set_kernel_hw(self, kernelh, kernelw):
        # for conv: filter/kernel: [filter_height, filter_width, in_channels, out_channels]
        # for pool: should be the size in each dimension of the input  - so [1, h, w, 1]
        self.kernel_size = [1, kernelh, kernelw, 1]

    def set_filter(self, some_l):
        self.filter_size = some_l

    def set_filter_hw_inout(self, h, w, in_ch, out_ch):
        self.filter_size = [h, w, in_ch, out_ch]

    def set_stride_hw(self, strideh, stridew):
        # strides = [1, stride, stride, 1]
        self.strides = [1, strideh, stridew, 1]

    def set_stride(self, some_l):
        self.strides = some_l

    def set_padding(self, padding):
        self.padding = padding

    def set_filter(self, some_l):
        # filter/kernel: [filter_height, filter_width, in_channels, out_channels]
        self.filter = tf.Variable(tf.random_normal(some_l), name="filterW" + str(self.node_num),
                                  dtype=tf.float32)

    def execute(self, some_op):
        self.node_num += 1
        method_name = 'execute_' + some_op
        try:
            method = getattr(self, method_name)
        except AttributeError:
            print method_name, "not found"
        else:
            return method()

    def execute_avg_pool(self):
        return tf.nn.avg_pool(self.inp, ksize=self.kernel_size, strides=self.strides, padding=self.padding,
                              name="avg_pool" + str(self.node_num))

    def execute_avg_pool3d(self):
        return tf.nn.avg_pool3d(self.inp, ksize=self.kernel_size, strides=self.strides, padding=self.padding,
                                name="avgpool3d" + str(self.node_num))

    def execute_conv2d(self):
        return tf.nn.conv2d(self.inp, self.filter, self.strides, self.padding, name="conv2d" + str(self.node_num))

    def execute_conv3d(self):
        return tf.nn.conv3d(self.inp, self.filter, self.strides, self.padding, name="conv3d" + str(self.node_num))

    def execute_max_pool(self):
        return tf.nn.max_pool(self.inp, self.kernel_size, self.strides, self.padding,
                              name="max_pool" + str(self.node_num))

    def flatten_convolution(self, tensor_in):
        tensor_in_shape = tensor_in.get_shape()
        tensor_in_flat = tf.reshape(tensor_in, [tensor_in_shape[0].value or -1, np.prod(tensor_in_shape[1:]).value])
        return tensor_in_flat
