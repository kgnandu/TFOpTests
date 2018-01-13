import tensorflow as tf
import numpy as np
from tensorflow.python.tools import freeze_graph

'''
Clean this up!!
 - reorg into a class maybe? (save_dir should be broken up as a basedir and a modeldir??)
 - add doc string
 - add links
 - credit where freeze_graph comes from
'''
base_dir = "/home/agibsonccc/pbdirs"

'''
Clean this up some day to one class that just takes filename with path and array
'''


def save_input(nparray, varname, save_dir):
    contentfile = base_dir + "/" + save_dir + "/" + varname + ".placeholder.csv"
    shapefile = base_dir + "/" + save_dir + "/" + varname + ".placeholder.shape"
    if (np.isscalar(nparray)):
        np.savetxt(shapefile, np.asarray([0]), fmt="%i")
        f = open(contentfile, 'w')
        f.write('{}'.format(nparray))
        f.close()
    else:
        np.savetxt(shapefile, np.asarray(nparray.shape), fmt="%i")
        np.savetxt(contentfile, np.ndarray.flatten(nparray), fmt="%10.8f")


def save_intermediate(nparray, varname, save_dir):
    contentfile = base_dir + "/" + save_dir + "/" + varname + ".prediction_inbw.csv"
    shapefile = base_dir + "/" + save_dir + "/" + varname + ".prediction_inbw.shape"
    if (np.isscalar(nparray)):
        np.savetxt(shapefile, np.asarray([0]), fmt="%i")
        f = open(contentfile, 'w')
        f.write('{}'.format(nparray))
        f.close()
    else:
        np.savetxt(shapefile, np.asarray(nparray.shape), fmt="%i")
        np.savetxt(contentfile, np.ndarray.flatten(nparray), fmt="%10.8f")


def save_graph(sess, all_saver, save_dir):
    all_saver.save(sess, base_dir + "/" + save_dir + "/data-all", global_step=1000)
    tf.train.write_graph(sess.graph_def, base_dir + "/" + save_dir, "model.txt", True)


def freeze_n_save_graph(save_dir):
    checkpoint = tf.train.get_checkpoint_state(base_dir + "/" + save_dir)
    input_checkpoint = checkpoint.model_checkpoint_path
    print input_checkpoint
    output_graph = base_dir + "/" + save_dir + "/frozen_model.pb"
    input_graph = base_dir + "/" + save_dir + "/model.txt"
    '''
    The below is just the same as running on the command line:
    python ~/anaconda2/lib/python2.7/site-packages/tensorflow/python/tools/freeze_graph.py --input_graph=model.txt --input_checkpoint=data-all-1000 --output_graph=frozen_graph.pb --output_node_name=output --input_binary=False
    Not on the command lines you have to give it all the defaults or it complains about "freeze_graph() takes at least 10 arguments"
    This might have been fixed in the 1.4 version of TF
    '''
    freeze_graph.freeze_graph(input_graph=input_graph,
                              input_saver="",
                              input_checkpoint=input_checkpoint,
                              output_graph=output_graph,
                              input_binary=False,
                              output_node_names="output",
                              restore_op_name="save/restore_all",
                              filename_tensor_name="save/Const:0",
                              clear_devices=True,
                              initializer_nodes="")

'''
def save_predictions(save_dir, outputs):
    for count,output in enumerate(outputs):
        contentfile = base_dir + "/" + save_dir + "/" + "output"+ .prediction.csv"
        shapefile = base_dir + "/" + save_dir + "/" + "output.prediction.shape"
        if (np.isscalar(output)):
            np.savetxt(shapefile, np.asarray([0]), fmt="%i")
            f = open(contentfile, 'w')
            f.write('{}'.format(output))
            f.close()
        else:
            np.savetxt(shapefile, np.asarray(output.shape), fmt="%i")
            np.savetxt(contentfile, np.ndarray.flatten(output), fmt="%10.8f")
'''


def save_prediction(save_dir, output):
    contentfile = base_dir + "/" + save_dir + "/" + "output.prediction.csv"
    shapefile = base_dir + "/" + save_dir + "/" + "output.prediction.shape"
    if (np.isscalar(output)):
        np.savetxt(shapefile, np.asarray([0]), fmt="%i")
        f = open(contentfile, 'w')
        f.write('{}'.format(output))
        f.close()
    else:
        np.savetxt(shapefile, np.asarray(output.shape), fmt="%i")
        np.savetxt(contentfile, np.ndarray.flatten(output), fmt="%10.8f")


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


def save_intermediate_nodes(save_dir, input_dict):
    graph = load_frozen_graph(save_dir)
    placeholder_dict = {}
    for op in graph.get_operations():
        if op.type != "Placeholder":
            continue
        # there is a prefix and a suffix - there should only be one prefix with import
        print op.name
        placeholder_name = "/".join(op.name.split("/")[1:])
        print input_dict[placeholder_name]
        placeholder_dict[op.name + ":0"] = input_dict[placeholder_name]

    for op in graph.get_operations():
        if op.type == "Placeholder":
            continue
        print("=====")
        print op.name
        if "while" in op.name:
            print ("SKIPPING")
            continue
        output_num = 0
        for op_output in op.outputs:
            print op_output.name
            with tf.Session(graph=graph) as sess:
                op_prediction = sess.run(op_output, feed_dict=placeholder_dict)
                save_to = ".".join(["____".join(op_output.name.split("/")[1:]).split(":")[0], str(output_num)])
                save_intermediate(op_prediction, save_to, save_dir)
                print op_prediction
                print("=====")
            output_num += 1
