import numpy as np
import tensorflow as tf

from tfoptests import persistor


def load_graph(model_file):
    graph = tf.Graph()
    graph_def = tf.GraphDef()

    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)

    return graph


model_file = '/Users/susaneraly/SKYMIND/nd4j/nd4j-backends/nd4j-tests/src/test/resources/tf_graphs/examples/conv_0/frozen_model.pb'
graph = load_graph(model_file)
for op in graph.get_operations():
    print(op.name)

'''
n = graph.get_operation_by_name('import/output')
#a = graph.get_tensor_by_name('import/conv2d0_w:0')
# = graph.get_tensor_by_name('import/conv2d2_b:0')

print("====")
#print(a)
#print(b)
print(n.get_attr('value'))
print "===="
'''

x = graph.get_tensor_by_name('import/input_0:0')
y = graph.get_tensor_by_name('import/output:0')
with tf.Session(graph=graph) as sess:
    np.random.seed(13)
    imsize = [4, 28, 28, 3]
    input_0 = np.random.uniform(size=imsize)
    # Note: we don't nee to initialize/restore anything
    # There is no Variables in this graph, only hardcoded constants
    prediction = sess.run(y, feed_dict={x: input_0})
    persistor.save_prediction("/Users/susaneraly/", prediction)
