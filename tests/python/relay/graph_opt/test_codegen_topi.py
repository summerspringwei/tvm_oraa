import numpy as np

import tvm
from tvm import relay, te, topi, tir
from tvm import IRModule
from tvm import meta_schedule as ms
from tvm.target import Target
from tvm.topi.utils import get_const_tuple
from tvm.relay.backend.contrib.oraa import op
from tvm.relay.backend.contrib.oraa.topi import pixel_shuffle_cuda
from tvm.relay.backend.contrib.oraa.tir.tensor_intrin import oraa_cuda
from tvm.script import tir as T

# we need to support the following conv:
# conv2d:
# 1x1, stride=1,padding=0,dilation=1
# 3x3, stride=1,padding=1,dilation=1
# 5x5  stride=1,padding=2,dilation=1
# depthwise_conv2d:
# 1x1, stride=1,padding=0,dilation=1
# 3x3, stride=1,padding=1,dilation=1
# 5x5  stride=1,padding=2,dilation=1 
def test_tensorize_topi_conv2d_1x1(input_shape:list,weight_shape:list):
    input = te.placeholder(input_shape, dtype="int8", name="input")
    n,c,h,w = input_shape
    oc,ic,kh,kw = weight_shape
    assert(ic==c)
    assert(kh==kw==1)
    weight = te.placeholder(weight_shape,dtype="int8", name="weight")
    out = topi.nn.conv2d_nchw(input, weight,stride=1,padding=0,dilation=1)
    func = te.create_prim_func([input, weight, out])
    ir_module_from_te = IRModule({"main": func})
    sch = tir.Schedule(ir_module_from_te)
    print(sch.mod)
    
def test_tensorize_topi_depthwise_conv2d_1x1(input_shape:list):
    input = te.placeholder(input_shape, dtype="int8", name="input")
    n,c,h,w = input_shape
    #(c, m, kh, kw)
    weight = te.placeholder((c,1,1,1),dtype="int8", name="weight")
    out = topi.nn.depthwise_conv2d_nchw(input,weight,1,0,1)
    func = te.create_prim_func([input, weight, out])
    ir_module_from_te = IRModule({"main": func})
    sch = tir.Schedule(ir_module_from_te)
    print(sch.mod)
    


def test_tensorize_topi_add(input_shape: list):
    in0 = te.placeholder(input_shape, dtype="int8", name="in0")
    in1 = te.placeholder(input_shape, dtype="int8", name="in1")
    out = topi.add(in0, in1)
    func = te.create_prim_func([in0, in1, out])
    ir_module_from_te = IRModule({"main": func})
    sch = tir.Schedule(ir_module_from_te)
    block = sch.get_block("T_add")
    (n, c, h, w) = sch.get_loops(block)
    no, ni = sch.split(n, factors=[None, 2])
    co, ci = sch.split(c, factors=[None, 8])
    ho, hi = sch.split(h, factors=[None, 4])
    wo, wi = sch.split(w, factors=[None, 4])
    sch.reorder(no, co, ho, wo, ni, ci, hi, wi)
    block_inner = sch.blockize(ni)
    a_shared = sch.cache_read(block_inner, 0, "shared")
    b_shared = sch.cache_read(block_inner, 1, "shared")
    e_shared = sch.cache_write(block_inner, 0, "shared")
    sch.compute_at(a_shared, wo)
    sch.compute_at(b_shared, wo)
    sch.reverse_compute_at(e_shared, wo)
    ani, _, _, _ = sch.get_loops(a_shared)[-4:]
    sch.tensorize(ani, oraa_cuda.ORAA_LDG2S_N2C8H4W4_INT8_INTRIN)
    bni, _, _, _ = sch.get_loops(b_shared)[-4:]
    sch.tensorize(bni, oraa_cuda.ORAA_LDG2S_N2C8H4W4_INT8_INTRIN)
    sch.tensorize(ni, oraa_cuda.ORAA_Add2_N2C8H4W4_INTRIN)
    e_n, _, _, _ = sch.get_loops(e_shared)[-4:]
    sch.tensorize(e_n, oraa_cuda.ORAA_STS2G_N2C8H4W4_INT8_INTRIN)
    sch.bind(no, "blockIdx.x")
    target = Target("oraa")
    func = tvm.build(sch.mod, [in0, in1, out], target=target)
    print(func)
    print(func.imported_modules[0].get_source())


"""
bug:
only support c,h,w = ic,ih,iw now. When blockize
(4,16,8,8)->(2,2,2,2|2,8,4,4)
but it generated(2,2,2,2|2,14,4,4)
"""
def test_tensorize_topi_pixel_shuffle(input_shape: list):
    in0 = te.placeholder(input_shape, dtype="int8", name="in0")
    out = topi.nn.depth_to_space(in0, 2)
    func = te.create_prim_func([in0, out])
    ir_module_from_te = IRModule({"main": func})
    sch = tir.Schedule(ir_module_from_te)
    print(sch.mod)
    # out2 = pixel_shuffle_cuda.pixel_shuffle_nchw(in0, [2,2])
    # func2 = te.create_prim_func([in0, out2])
    # ir_module_from_te2 = IRModule({"main": func})
    # sch2 = tir.Schedule(ir_module_from_te2)
    # print(sch2.mod)
    block = sch.get_block("depth_to_space")
    (n, c, h, w) = sch.get_loops(block)
    no, ni = sch.split(n, factors=[None, 2])
    co, ci = sch.split(c, factors=[None, 2])
    ho, hi = sch.split(h, factors=[None, 8])
    wo, wi = sch.split(w, factors=[None, 8])
    sch.reorder(no, co, ho, wo, ni, ci, hi, wi)
    print(sch.mod)
    block_inner = sch.blockize(ni)
    print(sch.mod)
    a_shared = sch.cache_read(block_inner, 0, "shared")
    e_shared = sch.cache_write(block_inner, 0, "shared")
    sch.compute_at(a_shared, wo)
    sch.reverse_compute_at(e_shared, wo)
    ani, _, _, _ = sch.get_loops(a_shared)[-4:]
    print(sch.mod)
    sch.tensorize(ani, oraa_cuda.ORAA_LDG2S_N2C8H4W4_INT8_INTRIN)
    sch.tensorize(ni, oraa_cuda.ORAA_PIXEL_SHUFFLE_N2C8H4W4_INTRIN)
    e_n, _, _, _ = sch.get_loops(e_shared)[-4:]
    sch.tensorize(e_n, oraa_cuda.ORAA_STS2G_N2C2H8W8_INT8_INTRIN)
    sch.bind(no, "blockIdx.x")
    target = Target("oraa")
    func = tvm.build(sch.mod, [in0, out], target=target)
    print(func)
    print(func.imported_modules[0].get_source())


if __name__ == "__main__":
    # tensorized
    # test_tensorize_topi_add((2,16,8,8))
    # test_tensorize_topi_pixel_shuffle((4, 16, 8, 8))
    # haven't tensorized
    test_tensorize_topi_conv2d_1x1((2,4,16,16),(32,4,1,1))
    test_tensorize_topi_depthwise_conv2d_1x1((2,8,16,16))
