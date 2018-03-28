# TFOpTests - Generate, persist and load tensorflow graphs for tests [![Build Status](https://travis-ci.org/deeplearning4j/TFOpTests.svg?branch=master)]

## Setup

To get started with this project first clone the DL4J test resources repository into
a folder of your choice:
```bash
git clone https://github.com/deeplearning4j/dl4j-test-resources
cd dl4j-test-resources
```

Next, put `DL4DL4J_TEST_RESOURCES` on your path, for instance by adding it to your `.zshrc` (or `.bashrc` etc.):

```bash
echo "export DL4J_TEST_RESOURCES=$(pwd)" >> $HOME/.zshrc
```

This is the path used in this project to generate test resources. If you wish to store results in
another path, just change the above resource path accordingly.

Finally, install this library locally by running `python setup.py develop`. It is recommended to
work with a Python virtual environment.

## Usage

To run a single test, for instance for a simple MLP with non-trivial bias terms in layers, run the following:

```python
python tests/mlp/test_bias_add.py
```

To generate all test resources, simply run:
```python
python -m pytest
```

## Adding new tests

The base for adding any new tests is extending a so called `TestGraph`, which is defined in `tfoptests.test_graph`. You will have to override functionality, as suitable, for all the methods except the `get_test_data` and the `get_placeholder` methods:

```python
    def get_placeholder_input(self, name):
        '''Get input tensor for given node name'''
        return None

    def _get_placeholder_shape(self, name):
        '''Get input tensor shape for given node name'''
        return None

    def list_inputs(self):
        '''List names of input nodes'''
        return ["input"]
```

These methods specify the input data and its shape for the graph we want to run, persist and test.

To give an example, to set up inputs and placeholders for a simple MLP:

```python
import numpy as np
import tensorflow as tf
from tfoptests.persistor import TensorFlowPersistor
from tfoptests.test_graph import TestGraph

n_hidden_1 = 10
num_input = 5
mini_batch = 4
num_classes = 3


class VanillaMLP(TestGraph):
    def list_inputs(self):
        return ["input"]

    def get_placeholder_input(self, name):
        if name == "input":
            input_0 = np.random.uniform(size=(mini_batch, num_input))
            return input_0

    def _get_placeholder_shape(self, name):
        if name == "input":
            return [None, num_input]
```

And to test we add in functionality and ops as needed as shown below:

```python
def test_vanilla_mlp():
    #Extends from TestGraph
    vanilla_mlp = VanillaMLP(seed=1337)
    #input placeholder
    in_node = vanilla_mlp.get_placeholder("input")
    # Define model
    weights = dict(
        h1=tf.Variable(tf.random_normal([num_input, n_hidden_1], dtype=tf.float64),
                       name="l0W"),
        out=tf.Variable(tf.random_normal([n_hidden_1, num_classes], dtype=tf.float64),
                        name="l1W")
    )
    biases = dict(
        b1=tf.Variable(tf.random_normal([n_hidden_1], dtype=tf.float64), name="l0B"),
        out=tf.Variable(tf.random_normal([num_classes], dtype=tf.float64), name="l1B")
    )
    layer_1 = tf.nn.bias_add(tf.matmul(in_node, weights['h1']), biases['b1'], name="l0Preout")
    layer_1_post_actv = tf.abs(layer_1, name="l0Out")
    logits = tf.nn.bias_add(tf.matmul(layer_1_post_actv, weights['out']), biases['out'], name="l1PreOut")
    #output
    out_node = tf.nn.softmax(logits, name='output')

    #Add input placeholders to a list
    placeholders = [in_node]
    #Add network outptus to a list
    predictions = [out_node]

    # Run and persist and set the save_dir to the name of the directory to write contents too
    tfp = TensorFlowPersistor(save_dir="mlp_00")
    predictions = tfp \
        .set_placeholders(placeholders) \
        .set_output_tensors(predictions) \
        .set_test_data(vanilla_mlp.get_test_data()) \
        .build_save_frozen_graph()
```

Note the `TensorFlowPersistor` method call. 
Under the hood it will:
- persist the tf graph after freezing
- write graph inputs, graph outputs and intermediate node results
- run asserts to ensure that predictions before and after freezing are the same
etc.

Check out the implementation for more details. 

An extremely bare bones example can be found in [test_expand_dim](tests/mathops/test_expand_dim.py)

These graphs are then used in integration tests for our _tensorflow model import_ . Graphs are imported into samediff and checks are run to ensure the correctness of the libnd4j implementation and its mapping.
Checks on the java side can be found in the nd4j repository in this [package](https://github.com/deeplearning4j/nd4j/tree/master/nd4j-backends/nd4j-tests/src/test/java/org/nd4j/imports/TFGraphs)

## Contributing

If there are missing operations or architectures that need to be covered, make sure to file an issue or open a pull request.  
