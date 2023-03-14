import numpy as np

import tvm
from tvm import relay, te, tir
from tvm import IRModule
from tvm import meta_schedule as ms
from tvm.target import Target
from tvm.topi.utils import get_const_tuple
from tvm.relay.backend.contrib.oraa import op
from tvm.relay.backend.contrib.oraa.topi import pixel_shuffle_cuda
from tvm.relay.backend.contrib.oraa.tir.tensor_intrin import oraa_cuda


def test_tensorize_oraa_pixel_shuffle(input_shape: list):
    """Test tensorize of pixel shuffle computation"""
    input_tensor = te.placeholder(input_shape, dtype="int8", name="input_name")
    output_tensor = pixel_shuffle_cuda.pixel_shuffle_nchw(input_tensor, (2, 2))
    output_tensor = pixel_shuffle_cuda.pixel_shuffle_nchw(input_tensor, [2, 2])
    func = te.create_prim_func([input_tensor, output_tensor])
    ir_module_from_te = IRModule({"main": func})

    sch = tir.Schedule(ir_module_from_te)
    block_pixel_shuffle = sch.get_block("PixelShuffle")
    (n, c, h, w) = sch.get_loops(block_pixel_shuffle)
    no, ni = sch.split(n, factors=[None, 2])
    co, ci = sch.split(c, factors=[None, 2])
    ho, hi = sch.split(h, factors=[None, 8])
    wo, wi = sch.split(w, factors=[None, 8])
    sch.reorder(no, co, ho, wo, ni, ci, hi, wi)
    
    sch.tensorize(ni, oraa_cuda.ORAA_PIXEL_SHUFFLE_N2C8H4W4_INTRIN)
    block_tensorized_pixel_shuffle = sch.get_block("PixelShuffle_o")
    sch.cache_read(block_tensorized_pixel_shuffle, 0, "shared")
    sch.cache_write(block_tensorized_pixel_shuffle, 0, "shared")

    return sch


def test_oraa_build(input_shape: list):
    sch = test_tensorize_oraa_pixel_shuffle(input_shape)
    print(sch.mod.script())
    input_placeholder = te.placeholder(input_shape,
                                               dtype='int8',
                                               name='input_tvm')
    output_shape = input_shape
    output_placeholder = te.placeholder(output_shape,
                                        dtype='int8',
                                        name='output_tvm')
    target = Target("oraa")
    func = tvm.build(sch.mod, [input_placeholder, output_placeholder],
                             target=target)
    print(func)
    input_np = np.array(np.random.randn(*input_shape) * 128,
                        dtype=np.byte)
    output_np = np.array(np.zeros(output_shape), dtype=np.byte)
    input_tvm = tvm.nd.array(input_np, device=tvm.cuda(0))
    output_tvm = tvm.nd.array(output_np, device=tvm.cuda(0))
    # func(input_tvm, output_tvm)


if __name__=="__main__":
    # test_tensorize_oraa_pixel_shuffle((2, 16, 8, 8))
    test_oraa_build((2, 16, 8, 8))
