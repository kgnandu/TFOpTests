import tensorflow as tf
import numpy as np
from tensorflow.python.tools import freeze_graph

'''
 - add doc string
 - add links to frozen graph explanation
'''

base_dir = "/Users/susaneraly/SKYMIND/dl4j-test-resources/src/main/resources/tf_graphs/examples"


def save_input(nparray, var_name, save_dir):
    content_file = base_dir + "/" + save_dir + "/" + var_name + ".placeholder.csv"
    shape_file = base_dir + "/" + save_dir + "/" + var_name + ".placeholder.shape"
    _write_to_file(nparray, content_file, shape_file)


def _save_intermediate(nparray, var_name, save_dir):
    content_file = base_dir + "/" + save_dir + "/" + var_name + ".prediction_inbw.csv"
    shape_file = base_dir + "/" + save_dir + "/" + var_name + ".prediction_inbw.shape"
    _write_to_file(nparray, content_file, shape_file)


def save_prediction(save_dir, output):
    content_file = base_dir + "/" + save_dir + "/" + "output.prediction.csv"
    shape_file = base_dir + "/" + save_dir + "/" + "output.prediction.shape"
    _write_to_file(output, content_file, shape_file)


def save_predictions(save_dir, output_dict):
    for output_name, output_value in output_dict.items():
        content_file = base_dir + "/" + save_dir + "/" + output_name + ".prediction.csv"
        shape_file = base_dir + "/" + save_dir + "/" + output_name + ".prediction.shape"
        _write_to_file(output_value, content_file, shape_file)


def save_graph(sess, all_saver, save_dir):
    all_saver.save(sess, base_dir + "/" + save_dir + "/data-all", global_step=1000)
    tf.train.write_graph(sess.graph_def, base_dir + "/" + save_dir, "model.txt", True)


def freeze_n_save_graph(save_dir, output_node_names):
    checkpoint = tf.train.get_checkpoint_state(base_dir + "/" + save_dir)
    input_checkpoint = checkpoint.model_checkpoint_path
    print input_checkpoint
    output_graph = base_dir + "/" + save_dir + "/frozen_model.pb"
    input_graph = base_dir + "/" + save_dir + "/model.txt"
    # hard coded values below are defaults
    freeze_graph.freeze_graph(input_graph=input_graph,
                              input_saver="",
                              input_checkpoint=input_checkpoint,
                              output_graph=output_graph,
                              input_binary=False,
                              output_node_names=output_node_names,
                              restore_op_name="save/restore_all",
                              filename_tensor_name="save/Const:0",
                              clear_devices=True,
                              initializer_nodes="")


def write_frozen_graph_txt(save_dir):
    graph_filename = base_dir + "/" + save_dir + "/frozen_model.pb"
    with tf.gfile.GFile(graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.train.write_graph(graph_def, base_dir + "/" + save_dir + '/', 'frozen_graph.pbtxt', True)


def load_frozen_graph(save_dir):
    graph_filename = base_dir + "/" + save_dir + "/frozen_model.pb"
    graph = tf.Graph()
    with tf.gfile.GFile(graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)
    return graph


def load_external_graph(model_file):
    graph = tf.Graph()
    graph_def = tf.GraphDef()

    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)

    return graph


def save_intermediate_nodes(save_dir, input_dict, please_print=True):
    graph = load_frozen_graph(save_dir)
    placeholder_dict = {}
    for op in graph.get_operations():
        if op.type != "Placeholder":
            continue
        # there is a prefix and a suffix - there should only be one prefix with import
        if please_print: print op.name
        placeholder_name = "/".join(op.name.split("/")[1:])
        if please_print: print input_dict[placeholder_name]
        placeholder_dict[op.name + ":0"] = input_dict[placeholder_name]

    if please_print: print("-----------------------------------------------------")
    for op in graph.get_operations():
        if op.type == "Placeholder":
            continue
        if please_print: print op.name
        if please_print: print op.type
        output_num = 0
        for op_output in op.outputs:
            if please_print: print op_output.name
            with tf.Session(graph=graph) as sess:
                try:
                    op_prediction = sess.run(op_output, feed_dict=placeholder_dict)
                    save_to = ".".join(["____".join(op_output.name.split("/")[1:]).split(":")[0], str(output_num)])
                    _save_intermediate(op_prediction, save_to, save_dir)
                    if please_print: print op_prediction
                    if please_print: print("-----------------------------------------------------")
                except:
                    if please_print: print("SKIPPING")
                    if please_print: print("-----------------------------------------------------")
            output_num += 1


def _write_to_file(nparray, content_file, shape_file):
    if np.isscalar(nparray):
        np.savetxt(shape_file, np.asarray([0]), fmt="%i")
        f = open(content_file, 'w')
        f.write('{}'.format(nparray))
        f.close()
    else:
        np.savetxt(shape_file, np.asarray(nparray.shape), fmt="%i")
        np.savetxt(content_file, np.ndarray.flatten(nparray), fmt="%10.8f")
