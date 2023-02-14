"""Create frozen TF models for testing the graph optimization of Open Research AI Architecture"""

import os

import tensorflow as tf
from tvm import relay


import tf_utils


def tf_addn_single_op(input_list: list):
    R"""Create a TF model with a single AddN operator"""
    return tf.math.add_n(input_list)


def freeze_tf_addn_single_op(input_shape: list,
                             num_inputs: int,
                             dtype=tf.float32):
    r"""Freeze a TF model with a single AddN operator

    Parameters
    ----------
    input_shape: the input shape with NCHW format
    num_inputs: the number of inputs, corresponding to the `N` in `AddN`

    Returns:
    the file path of the frozen TF model
    """

    input_list = [
        tf.keras.Input(shape=input_shape,
                       name="input_{}".format(i),
                       dtype=dtype) for i in range(num_inputs)
    ]
    output = tf_addn_single_op(input_list)
    model = tf.keras.Model(inputs=input_list, outputs=[output])
    folder_path = os.path.join("/tmp", "intel_tvm/models/")
    input_shape_str = [str(i) for i in input_shape]
    model_name = "{}_n{}_s{}".format(tf_addn_single_op.__name__,
                                   num_inputs,
                                   ",".join(input_shape_str))
    return tf_utils.tf_freeze_keras_model(model, folder_path, model_name)


def tf_reshape_transpose_reshape(x, input_shape, upscale_factor: int):
    r"""Implementation of PixelShuffle using reshape and transpose operators

    Rearranges elements in a tensor of shape (N, C*R*R, H, W) to a tensor of shape (N, C, H*R, W*R),
    where R is an upscale_factor.

    Parameters
    --------
    x: the input tensor
    upscale_factor: factor to increase spatial resolution by

    Returns:
    ----------
    The output tensor
    """
    n, c, h, w = input_shape
    x = tf.reshape(
        x, [n, c // (upscale_factor * upscale_factor), upscale_factor, upscale_factor, h, w])
    x = tf.transpose(x, [0, 1, 4, 2, 5, 3])
    x = tf.reshape(
        x, [n, c // (upscale_factor * upscale_factor), h * upscale_factor, w * upscale_factor])

    return x


def freeze_tf_reshape_transpose_reshape(input_shape,
                                        upscale_factor,
                                        dtype=tf.float32):
    """Freeze a TF model with the following structure:
    reshape
       |
    transpose
       |
    reshape

    Parameters:
    ----------
    input_shape: The input tensor's shape
    upscale_factor: factor to increase spatial resolution by

    Returns:
    the file path of the frozen TF model
    """
    input_list = [tf.keras.Input(shape=input_shape, name="input", dtype=dtype)]
    output = tf_reshape_transpose_reshape(input_list[0], input_shape,
                                          upscale_factor)
    model = tf.keras.Model(inputs=input_list, outputs=[output])
    folder_path = os.path.join("/tmp", "intel_tvm/models/")
    input_shape_str = [str(i) for i in input_shape]
    model_name = "{}_s{}".format(tf_reshape_transpose_reshape.__name__,
                                 ",".join(input_shape_str))
    tf_utils.tf_freeze_keras_model(model, folder_path, model_name)

    return os.path.join(folder_path, model_name + ".pb")


def load_tf_to_relay(model_file):
    r"""Load the frozen TF model and convert to relay module"""
    with tf.io.gfile.GFile(model_file, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
        mod, _ = relay.frontend.from_tensorflow(graph_def)
        graph = mod["main"]
        print(graph)

        return mod


if __name__ == "__main__":
    load_tf_to_relay(freeze_tf_reshape_transpose_reshape([1, 16, 56, 56], 2))
    load_tf_to_relay(freeze_tf_addn_single_op([
        1,
        32,
        56,
        56,
    ], 5))
