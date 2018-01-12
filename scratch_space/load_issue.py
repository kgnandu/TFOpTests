import numpy as np
import tensorflow as tf

from graphs.mathops.non2d_1 import get_input


def load_graph(model_file):
    graph = tf.Graph()
    graph_def = tf.GraphDef()

    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)

    return graph


model_file = '/Users/susaneraly/Desktop/darkflow/built_graph/yolo.pb'
graph = load_graph(model_file)
for op in graph.get_operations():
    #print (op.name)
    if op.type == "Placeholder":
        print (op.name)
        print("is placeholder")


with tf.gfile.GFile(model_file, "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    #tf.train.write_graph(graph_def, '/Users/susaneraly/Desktop/darkflow/built_graph/','frozen_graph.pbtxt', True)

'''
#a = graph.get_tensor_by_name('import/conv2d0_w:0')
# = graph.get_tensor_by_name('import/conv2d2_b:0')

print("====")
#print(a)
#print(b)
print(n.get_attr('value'))
print "===="
'''

x = graph.get_tensor_by_name('import/input:0')
y = graph.get_tensor_by_name('import/output:0')
with tf.Session(graph=graph) as sess:
    # Note: we don't nee to initialize/restore anything
    # There is no Variables in this graph, only hardcoded constants
    prediction = sess.run(y, feed_dict={x: np.random.random((1,608,608,3))})
    print(prediction)
    print(prediction.shape)
