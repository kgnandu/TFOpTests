import numpy as np
import tensorflow as tf

def load_graph(model_file):
    graph = tf.Graph()
    graph_def = tf.GraphDef()

    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)

    return graph


model_file = "/Users/susaneraly/SKYMIND/nd4j/nd4j-backends/nd4j-tests/src/test/resources/tf_graphs/tensorflow_inception_graph.pb"
graph = load_graph(model_file)
for op in graph.get_operations():
    print (op.name)


n = graph.get_operation_by_name('import/conv2d0_w')
a = graph.get_tensor_by_name('import/conv2d0_w:0')
b = graph.get_tensor_by_name('import/conv2d2_b:0')

print("====")
print(a)
print(b)
print(n.get_attr('value'))
print "===="
