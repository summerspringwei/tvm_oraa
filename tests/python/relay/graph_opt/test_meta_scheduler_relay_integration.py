# pylint: disable=invalid-name
import logging
import sys
import tempfile

import numpy as np
import torch

import tvm
from tvm import relay, te, tir
from tvm import IRModule
from tvm import meta_schedule as ms
from tvm.target import Target
from tvm.topi.utils import get_const_tuple
from tvm.relay.backend.contrib.oraa import op
from tvm.relay.backend.contrib.oraa.topi import pixel_shuffle_cuda
from tvm.relay.backend.contrib.oraa.tir.tensor_intrin import oraa_cuda

logging.getLogger("te_compiler").setLevel(logging.INFO)
logging.getLogger("te_compiler").addHandler(logging.StreamHandler(sys.stdout))


def workload():
    """Create a relay function contains reshape-transpose-reshape"""
    x = relay.var("x", shape=(1, 16, 56, 56), dtype="int8")
    reshape_node = relay.reshape(x, [1, 16 / 4, 2, 2, 56, 56])
    transpose_node = relay.transpose(reshape_node, [0, 1, 4, 2, 5, 3])
    reshape_node_2 = relay.reshape(transpose_node, [1, 16 / 4, 56 * 2, 56 * 2])

    return relay.Function([x], reshape_node_2)


def test_meta_schedule_relay_integration_oraa_pixel_shuffle(input_shape: list):
    """Test using meta scheduler to search a schedule for relay module
    
    Parameters:
    ----------
    input_shape: list of int
    Set input tensor's shape
    """
    a = relay.var("a", shape=input_shape, dtype="int8")
    b = op.oraa_pixel_shuffle(a)
    N, C, H, W = input_shape
    output_shape = [N, C // 4, H * 2, W * 2]
    mod = IRModule({"main": relay.Function([a], b)})
    extracted_tasks = ms.relay_integration.extract_tasks(mod,
                                                         target="llvm",
                                                         params={})
    for task in extracted_tasks:
        print(task.mod)
        print(task.dispatched)

    oraa_pixel_shuffle_tir = extracted_tasks[0].dispatched[0]
    with tempfile.TemporaryDirectory() as work_dir:
        target = Target("nvidia/nvidia-a100")
        database = ms.tir_integration.tune_tir(
            oraa_pixel_shuffle_tir,
            target=target,
            work_dir=work_dir,
            max_trials_global=32,
            num_trials_per_iter=16,
        )
        sch = ms.tir_integration.compile_tir(database, oraa_pixel_shuffle_tir,
                                             target)
        if sch is None:
            print("No valid schedule found!")
        else:
            sch.mod.show()
            sch.trace.show()
            input_placeholder = te.placeholder(input_shape,
                                               dtype='int8',
                                               name='input_tvm')
            output_placeholder = te.placeholder(output_shape,
                                                dtype='int8',
                                                name='output_tvm')
            func = tvm.build(sch.mod, [input_placeholder, output_placeholder],
                             target=target)

            input_np = np.array(np.random.randn(*input_shape) * 128,
                                dtype=np.byte)
            output_np = np.array(np.zeros(output_shape), dtype=np.byte)
            input_tvm = tvm.nd.array(input_np, device=tvm.cuda(0))
            output_tvm = tvm.nd.array(output_np, device=tvm.cuda(0))
            func(input_tvm, output_tvm)
            output_torch = torch.pixel_shuffle(
                torch.tensor(input_np, dtype=torch.int8), 2)
            np.testing.assert_allclose(output_tvm.numpy(),
                                       output_torch.numpy())


if __name__ == "__main__":
    # test_meta_schedule_dynamic_loop_extent()
    # test_pixel_shuffle((1, 64, 28, 28))
    # test_pixel_shuffle((1, 4, 8, 8))
    # test_meta_schedule_relay_integration_oraa_pixel_shuffle((1, 4, 8, 8))
    pass
