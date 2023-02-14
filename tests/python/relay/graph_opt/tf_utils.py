"""A set of functions to freeze models to tensorflow"""

import os
import logging

import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

FORMAT = '%(asctime)s %(filename)s:%(lineno)d %(message)s'
logging.basicConfig(format=FORMAT, level=logging.INFO)

def tf_freeze_keras_model(keras_model,
                          frozen_out_path,
                          frozen_graph_filename) -> str:
    r"""Save tensorflow keras model as frozen pb file

    Parameters
    ----------
    keras_model : model created using keras api
    frozen_out_path : path of the directory where you want to save your model
    frozen_graph_filename : name of the frozen model file

    Returns
    ----------
    str: the frozen model file path
    """

    full_model = tf.function(lambda x: keras_model(x))
    spec_list = []
    for input in keras_model.inputs:
        spec_list.append(tf.TensorSpec(input.shape, input.dtype))
    full_model = full_model.get_concrete_function(tuple(spec_list))

    # Get frozen ConcreteFunction
    frozen_func = convert_variables_to_constants_v2(full_model)
    frozen_func.graph.as_graph_def()
    layers = [op.name for op in frozen_func.graph.get_operations()]
    logging.info("-" * 60)
    logging.info("Frozen model layers: ")
    for layer in layers:
        logging.info(layer)
    logging.info("-" * 60)
    logging.info("Frozen model inputs: ")
    logging.info(frozen_func.inputs)
    logging.info("Frozen model outputs: ")
    logging.info(frozen_func.outputs)
    # Save frozen graph to disk
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                      logdir=frozen_out_path,
                      name=f"{frozen_graph_filename}.pb",
                      as_text=False)
    # Save its text representation
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                      logdir=frozen_out_path,
                      name=f"{frozen_graph_filename}.pbtxt",
                      as_text=True)

    return os.path.join(frozen_out_path, frozen_graph_filename+".pb")


def shape_list_to_str(shape_list: list):
    shape_str = ",".join(shape_list)
    return shape_str
