import errno
import numpy as np
from tensorflow.python.ops import linalg_ops
import os

from tests.mathops.norm_00 import save_dir
from tfoptests import persistor

'''
import numpy as np
FIGURE THIS OUT!!
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for tensorflow.ops.tf.norm."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.platform import test as test_lib


def _AddTest(test, test_name, fn):
    test_name = "_".join(["test", test_name])
    print(test_name + "===========")
    if hasattr(test, test_name):
        raise RuntimeError("Test %s defined more than once" % test_name)
    setattr(test, test_name, fn)


class norm_00(test_lib.TestCase):
    def testBadOrder(self):
        matrix = [[0., 1.], [2., 3.]]
        for ord_ in "foo", -7, -1.1, 0:
            with self.assertRaisesRegexp(ValueError,
                                         "'ord' must be a supported vector norm"):
                linalg_ops.norm(matrix, ord="fro")

        for ord_ in "foo", -7, -1.1, 0:
            with self.assertRaisesRegexp(ValueError,
                                         "'ord' must be a supported vector norm"):
                linalg_ops.norm(matrix, ord=ord_, axis=-1)

        for ord_ in 1.1, 2:
            with self.assertRaisesRegexp(ValueError,
                                         "'ord' must be a supported matrix norm"):
                linalg_ops.norm(matrix, ord=ord_, axis=[-2, -1])

    def testInvalidAxis(self):
        matrix = [[0., 1.], [2., 3.]]
        for axis_ in [], [1, 2, 3], [[1]], [[1], [2]], [3.1415], [1, 1]:
            error_prefix = ("'axis' must be None, an integer, or a tuple of 2 unique "
                            "integers")
            with self.assertRaisesRegexp(ValueError, error_prefix):
                linalg_ops.norm(matrix, axis=axis_)


def _GetNormOpTest(dtype_, shape_, ord_, axis_, keep_dims_, use_static_shape_):
    def _CompareNorm(self, matrix):
        np_norm = np.linalg.norm(matrix, ord=ord_, axis=axis_, keepdims=keep_dims_)
        with self.test_session(use_gpu=True) as sess:
            if use_static_shape_:
                tf_matrix = constant_op.constant(matrix)
                tf_norm = linalg_ops.norm(
                    tf_matrix, ord=ord_, axis=axis_, keep_dims=keep_dims_)
                tf_norm_val = sess.run(tf_norm)
            else:
                tf_matrix = array_ops.placeholder(dtype_)
                tf_norm = linalg_ops.norm(
                    tf_matrix, ord=ord_, axis=axis_, keep_dims=keep_dims_)
                tf_norm_val = sess.run(tf_norm, feed_dict={tf_matrix: matrix})
        self.assertAllClose(np_norm, tf_norm_val)

    def Test(self):
        is_matrix_norm = (isinstance(axis_, tuple) or
                          isinstance(axis_, list)) and len(axis_) == 2
        is_fancy_p_norm = np.isreal(ord_) and np.floor(ord_) != ord_
        if ((not is_matrix_norm and ord_ == "fro") or
                (is_matrix_norm and is_fancy_p_norm)):
            self.skipTest("Not supported by neither numpy.linalg.norm nor tf.norm")
        if is_matrix_norm and ord_ == 2:
            self.skipTest("Not supported by tf.norm")
        if ord_ == 'euclidean' or (axis_ is None and len(shape) > 2):
            self.skipTest("Not supported by numpy.linalg.norm")
        matrix = np.random.randn(*shape_).astype(dtype_)
        if dtype_ in (np.complex64, np.complex128):
            matrix += 1j * np.random.randn(*shape_).astype(dtype_)
        _CompareNorm(self, matrix)

    return Test


# pylint: disable=redefined-builtin
if __name__ == "__main__":
    print("In main")
    for use_static_shape in False, True:
        for dtype in np.float32, np.float64, np.complex64, np.complex128:
            for rows in 2, 5:
                for cols in 2, 5:
                    for batch in [], [2], [2, 3]:
                        shape = batch + [rows, cols]
                        for ord in "euclidean", "fro", 0.5, 1, 2, np.inf:
                            for axis in [
                                None, (-2, -1), (-1, -2), -len(shape), 0, len(shape) - 1
                            ]:
                                for keep_dims in False, True:
                                    name = "%s_%s_ord_%s_axis_%s_%s_%s" % (
                                        dtype.__name__, "_".join(map(str, shape)), ord, axis,
                                        keep_dims, use_static_shape)
                                    _AddTest(norm_00, "Norm_" + name,
                                             _GetNormOpTest(dtype, shape, ord, axis, keep_dims,
                                                            use_static_shape))

    test_lib.main()
'''
import tensorflow as tf


def _GetNormOpTest(dtype_, shape_, ord_, axis_, keep_dims_, use_static_shape_, save_dir_):
    def _CompareNorm(matrix):
        # tf_matrix = tf.Variable(matrix,name="input")
        in0 = tf.Variable(tf.random_normal(matrix.shape), name="in0", dtype=tf.float32)
        tf_matrix = tf.placeholder("float", matrix.shape, name="input") + in0
        tf_norm = linalg_ops.norm(
            tf_matrix, ord=ord_, axis=axis_, keep_dims=keep_dims_, name="norm_op")
        output = tf.identity(tf_norm, name="output")
        init = tf.global_variables_initializer()
        all_saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(init)
            # if use_static_shape_:
            #    tf_matrix = constant_op.constant(matrix)
            #    tf_norm = linalg_ops.norm(
            #        tf_matrix, ord=ord_, axis=axis_, keep_dims=keep_dims_)
            #    tf_norm_val = sess.run(tf_norm)
            # else:
            prediction = sess.run(output, feed_dict={tf_matrix: matrix})
            persistor.save_input(matrix, "input", save_dir_)
            persistor.save_graph(sess, all_saver, save_dir_)
            persistor.save_prediction(save_dir_, prediction)
        persistor.freeze_n_save_graph(save_dir_)
        persistor.write_frozen_graph_txt(save_dir_)
        persistor.save_intermediate_nodes(save_dir_, {"input": matrix})

    def Test(save_dir_i_):
        is_matrix_norm = (isinstance(axis_, tuple) or
                          isinstance(axis_, list)) and len(axis_) == 2
        is_fancy_p_norm = np.isreal(ord_) and np.floor(ord_) != ord_
        if ((not is_matrix_norm and ord_ == "fro") or
                (is_matrix_norm and is_fancy_p_norm)):
            print("Not supported by neither numpy.linalg.norm nor tf.norm")
            print("==========================================================")
            return
        if is_matrix_norm and ord_ == 2:
            print("Not supported by tf.norm")
            print("==========================================================")
            return
        matrix = np.random.randn(*shape_).astype(dtype_)
        # TODO: Remove Susan's path.
        test_info = "/Users/susaneraly/SKYMIND/dl4j-test-resources/src/main/resources/tf_graphs/examples/" + \
            save_dir_i_ + "/test.info"
        print("writing to...")
        print(test_info)
        if not os.path.exists(os.path.dirname(test_info)):
            try:
                os.makedirs(os.path.dirname(test_info))
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise
        print("matrix dims")
        print(matrix.shape)
        with open(test_info, "w") as f:
            f.write("ord ")
            f.write(str(ord_))
            f.write("\naxis ")
            f.write(str(axis_))
            f.write("\nkeep_dims ")
            f.write(str(keep_dims_))
            f.write("\ninput matrix shape ")
            f.write(str(matrix.shape))
            if dtype_ in (np.complex64, np.complex128):
                matrix += 1j * np.random.randn(*shape_).astype(dtype_)
            _CompareNorm(matrix)
            print("==========================================================")

    return Test


use_static_shape = False
dtype = np.float32
test_num = 0
# for rows in 2, 5:
for rows in [2]:
    # for cols in 2, 5:
    for cols in [5]:
        # for batch in [], [2], [2, 3]:
        for batch in [[]]:
            shape = batch + [rows, cols]
            for ord in "euclidean", "fro", 0.5, 1, 2, np.inf:
                # for axis in [None, (-2, -1), (-1, -2), -len(shape), 0, len(shape) - 1]:
                for axis in [None, (-2, -1)]:
                    # for keep_dims in False, True:
                    for keep_dims in [False]:
                        name = "%s_ord_%s_axis_%s_%s" % (
                            "_".join(map(str, shape)), ord, axis,
                            keep_dims)
                        if name not in ["2_2_ord_0.5_axis_None_False",
                                        "2_2_ord_0.5_axis_None_True"]:
                            print(name)
                            save_dir_i = save_dir + "/" + "norm_" + str(test_num)
                            something = _GetNormOpTest(dtype, shape, ord, axis, keep_dims,
                                                       use_static_shape, save_dir_i)
                            something(save_dir_i)
                            test_num += 1
print("ALL DONE")
