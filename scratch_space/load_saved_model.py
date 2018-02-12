import tensorflow as tf
import numpy as np
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.tools import freeze_graph

base_dir = "/Users/susaneraly/SKYMIND/nd4j/nd4j-backends/nd4j-tests/src/test/resources/tf_graphs/examples"
model_name = "bias_add_test"
save_dir = base_dir + "/" + model_name
input_key = "input"
output_key = "output"

with tf.Session(graph=tf.Graph()) as sess:
    input_0 = np.linspace(1, 40, 40).reshape(10, 4)
    meta_graph_def = tf.saved_model.loader.load(sess, [tag_constants.SERVING], save_dir)
    # print meta_graph_def
    signature = meta_graph_def.signature_def
    print(signature)

    input_tensor_name = signature[tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].inputs[
        input_key].name
    output_tensor_name = signature[tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].outputs[
        output_key].name

    input_tensor = sess.graph.get_tensor_by_name(input_tensor_name)
    output_tensor = sess.graph.get_tensor_by_name(output_tensor_name)

    prediction = sess.run(output_tensor, feed_dict={input_tensor: input_0})
    print(prediction)

    input_graph_filename = "/Users/susaneraly/SKYMIND/nd4j/nd4j-backends/nd4j-tests/src/test/resources/tf_graphs/examples/bias_add_test/saved_model.pb"
    output_graph_filename = "/Users/susaneraly/SKYMIND/nd4j/nd4j-backends/nd4j-tests/src/test/resources/tf_graphs/examples/bias_add_test/frozen_model.pb"

    freeze_graph.freeze_graph(input_graph=input_graph_filename,
                              input_saver=save_dir,
                              input_checkpoint=None,
                              output_graph=output_graph_filename,
                              input_binary=False,
                              output_node_names=output_key,
                              restore_op_name=None,
                              filename_tensor_name=None,
                              clear_devices=True)
