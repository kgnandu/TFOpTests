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

The base for adding any new tests is extending a so called `TensorFlowPersistor` (TFP), which is defined in `tfoptests.persistor`. This abstract base class has two members you need to implement:

```python
@abstractmethod
def _get_input(self, name):
    '''Get input tensor for given node name'''
    raise NotImplementedError

@abstractmethod
def _get_input_shape(self, name):
    '''Get input tensor shape for given node name'''
    raise NotImplementedError
```

These two methods specify the input data and its shape for the graph we want to run and persist.

To give an example, to implement a persistor for an MLP with added bias terms, you define:

```python
import numpy as np
from tfoptests.persistor import TensorFlowPersistor as TFP


class BiasAdd(TFP):
    def _get_input(self, name):
        if name == "input":
            input_0 = np.linspace(1, 40, 40).reshape(10, 4)
            return input_0

    def _get_input_shape(self, name):
        if name == "input":
            return [None, 4]
```

Running the test is a simple four-step procedure:

```python
# 1. Initialize your TFP
tfp = BiasAdd(save_dir="bias_add", seed=1337)

# 2. Set input tensor
in_node = tfp.get_placeholder("input")
tfp.set_placeholders([in_node])

# 3. Set output tensor
biases = tf.Variable(tf.lin_space(1.0, 4.0, 4), name="bias")
out_node = tf.nn.bias_add(in_node, tf.cast(biases, dtype=tf.float64), name="output")
tfp.set_output_tensors([out_node])

# 4. Run and persist your tensorflow graph
tfp.run_and_save_graph()
```

Note that `run_and_save_graph` does a lot of things for you, such as persisting the tf graph, inputs, outputs and intermediate results. Check out the implementation for details.

## Contributing

Skymind.ai uses the output of these test results as integration tests for our _tensorflow model import_. If there are missing operations or architectures that need to be covered, make sure to file an issue or open a pull request.  
