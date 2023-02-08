# import pytest
import os
import logging

import tensorflow as tf
import tvm
from tvm import relay
from tvm.relay.dataflow_pattern import TupleGetItemPattern, is_op, wildcard
from tvm.relay.testing import run_opt_pass

import tf_utils

print(tf.__version__)

def tf_addn_single_op(input_list: list):
    return tf.math.add_n(input_list)


def freeze_tf_addn_single_op(input_shape: list, num_inputs: int, dtype=tf.float32):
    input_list = [tf.keras.Input(shape=input_shape, name="input_{}".format(i), dtype=dtype) for i in range(num_inputs)]
    output = tf_addn_single_op(input_list)
    model = tf.keras.Model(inputs=input_list, outputs=[output])
    folder_path = os.path.join("/tmp", "intel_tvm/models/")
    input_shape_str = [str(i) for i in input_shape]
    model_name = "_n{}_s{}".format(tf_addn_single_op.__name__, num_inputs, ",".join(input_shape_str))
    tf_utils.tf_freeze_keras_model(model, folder_path, model_name)


def tf_reshape_transpose_reshape(x, input_shape, tile_size: int):
    n, c, h, w = input_shape
    x = tf.reshape(x, [n, c//(tile_size*tile_size), tile_size, tile_size, h, w])
    x = tf.transpose(x, [0, 1, 4, 2, 5, 3])
    x = tf.reshape(x, [n, c // (tile_size*tile_size), h * tile_size, w * tile_size])

    return x


def freeze_tf_reshape_transpose_reshape(input_shape, tile_size, dtype=tf.float32):
    input_list = [tf.keras.Input(shape=input_shape, name="input", dtype=dtype)]
    output = tf_reshape_transpose_reshape(input_list[0], input_shape, tile_size)
    model = tf.keras.Model(inputs=input_list, outputs=[output])
    folder_path = os.path.join("/tmp", "intel_tvm/models/")
    input_shape_str = [str(i) for i in input_shape]
    model_name = "{}_s{}".format(tf_reshape_transpose_reshape.__name__, ",".join(input_shape_str))
    tf_utils.tf_freeze_keras_model(model, folder_path, model_name)

    return os.path.join(folder_path, model_name + ".pb")


def make_add_relu_pattern():
    r"""Create a pattern to match the following graph.

     add
      |
    relu
    """
    add_node = wildcard() + wildcard()
    r = is_op("nn.relu")(add_node)
    return r


def make_reshape_transpose_reshape_pattern():
    r"""Create a pattern to match the following graph.

    reshape
       |
    transpose
       |
    reshape
    """
    x, y, z = wildcard(), wildcard(), wildcard()
    # a = relay.var("a", shape=(1, 16, 56, 56))
    a = wildcard()
    # reshape_node = is_op("reshape")(a, x)
    # transpose_node = is_op("transpose")(reshape_node, y)
    # r = is_op("reshape")(transpose_node, z)

    # We only need to match tensor rather than other parameters here
    reshape_node = is_op("reshape")(a)
    transpose_node = is_op("transpose")(reshape_node)
    r = is_op("reshape")(transpose_node)

    # x, y, z, a = wildcard(), wildcard(), wildcard(), wildcard()
    # x = relay.var("a", shape=(1, 16, 56, 56))
    # reshape_node = is_op("reshape")(x, [1, 16/4, 2, 2, 56, 56])
    # transpose_node = is_op("transpose")(reshape_node, [0, 1, 4, 2, 5, 3])
    # r = is_op("reshape")(transpose_node, [1, 16/4, 56*2, 56*2])
    return r


def relay_reshape_transpose_reshape():
    a = relay.var("a", shape=(1, 16, 56, 56))
    reshape_node = relay.reshape(a, [1, 16/4, 2, 2, 56, 56])
    transpose_node = relay.transpose(reshape_node, [0, 1, 4, 2, 5, 3])
    reshape_node_2 = relay.reshape(transpose_node, [1, 16/4, 56*2, 56*2])
    
    return relay.Function([a], reshape_node_2)


def load_tf_to_relay(model_file):
    print(model_file)
    with tf.io.gfile.GFile(model_file, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
        # print(graph_def)
        # layers = [op.name for op in graph_def.get_operations()]
        mod, params = relay.frontend.from_tensorflow(graph_def)
        graph = mod["main"]
        print(graph)

        pattern_table = [
          ("pixel_shuffle", make_reshape_transpose_reshape_pattern()),
        ]
        result = run_opt_pass(
          graph, relay.transform.MergeComposite(pattern_table), import_prelude=False
        )
        print(result)


def test_reshape_transpose_reshape():
    graph = relay_reshape_transpose_reshape()
    print(graph)
    pattern_table = [
          ("pixel_shuffle", make_reshape_transpose_reshape_pattern()),
        ]
    result = run_opt_pass(
          graph, relay.transform.MergeComposite(pattern_table), import_prelude=False
    )
    print(result)
    



if __name__=="__main__":
    # freeze_tf_addn_single_op([56,56,16,16], 5)
    file_path = freeze_tf_reshape_transpose_reshape([1,16,56,56], 2)
    load_tf_to_relay(file_path)
    # test_reshape_transpose_reshape()
    # test_simple_merge()
   
