import numpy as np
import tensorflow as tf

from graphs.mathops.non2d_1 import get_input
from helper.load_save_utils import load_external_graph

# model_file = '/Users/susaneraly/Desktop/darkflow/built_graph/yolo.pb'
# model_file = '/Users/susaneraly/SKYMIND/dl4j-test-resources/src/main/resources/tf_graphs/examples/ssd_mobilenet_v1_coco/frozen_model.pb'
model_file = '/Users/susaneraly/SKYMIND/dl4j-test-resources/src/main/resources/tf_graphs/examples/yolov2_608x608/frozen_model.pb'
graph = load_external_graph(model_file)
for op in graph.get_operations():
    print (op.name)
    if op.type == "Placeholder":
        print (op.name)
        print("is placeholder")

with tf.gfile.GFile(model_file, "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    # tf.train.write_graph(graph_def, '/Users/susaneraly/Desktop/darkflow/built_graph/','frozen_graph.pbtxt', True)

print("====")
# print(a)
# print(b)
# print(n.get_attr('value'))
# print "===="

x = graph.get_tensor_by_name('import/input:0')
# x = graph.get_tensor_by_name('import/image_tensor:0')
y = graph.get_tensor_by_name('import/output:0')
# y = graph.get_tensor_by_name('import/detection_classes:0')
with tf.Session(graph=graph) as sess:
    # Note: we don't nee to initialize/restore anything
    # There is no Variables in this graph, only hardcoded constants
    prediction = sess.run(y, feed_dict={x: np.random.random((4, 608, 608, 3))})
    print(prediction)
    print(prediction.shape)
