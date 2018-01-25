import numpy as np
import tensorflow as tf

from helper import load_save_utils
from helper.load_save_utils import load_external_graph
from model_zoo.yolov2_608x608 import model_file, save_dir

graph = load_external_graph(model_file)

x = graph.get_tensor_by_name('import/input:0')
y = graph.get_tensor_by_name('import/output:0')
with tf.Session(graph=graph) as sess:
    # Note: we don't nee to initialize/restore anything
    # There is no Variables in this graph, only hardcoded constants
    input_0 = np.random.random((1, 608, 608, 3))
    prediction = sess.run(y, feed_dict={x: input_0})
    print(prediction)
    print(prediction.shape)
    load_save_utils.save_input(input_0, "input", save_dir)
    load_save_utils.save_prediction(save_dir, prediction)
    load_save_utils.write_frozen_graph_txt(save_dir)
