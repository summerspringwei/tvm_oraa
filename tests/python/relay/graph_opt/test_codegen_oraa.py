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
from tvm.script import tir as T
import numpy as np


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
    block_inner = sch.blockize(ni)
    a_shared = sch.cache_read(block_inner, 0, "shared")
    e_shared = sch.cache_write(block_inner, 0, "shared")
    sch.compute_at(a_shared, wo)
    sch.reverse_compute_at(e_shared, wo)
    ani, _, _, _ = sch.get_loops(a_shared)[-4:]
    sch.tensorize(ani, oraa_cuda.ORAA_LDG2S_N2C8H4W4_INT8_INTRIN)
    sch.tensorize(ni, oraa_cuda.ORAA_PIXEL_SHUFFLE_N2C8H4W4_INTRIN)
    e_n, _, _, _ = sch.get_loops(e_shared)[-4:]
    sch.tensorize(e_n, oraa_cuda.ORAA_STS2G_N2C2H8W8_INT8_INTRIN)
    sch.bind(no, "blockIdx.x")
    # print(sch.mod)
    target = Target("oraa")
    # target = 'cuda'
    func = tvm.build(sch.mod, [input_tensor, output_tensor], target=target)
    print(func)
    print(func.imported_modules[0].get_source())


def test_tensorize_oraa_pixel_unshuffle(input_shape: list):
    """Test tensorize of pixel unshuffle computation"""
    input_tensor = te.placeholder(input_shape, dtype="int8", name="input_name")
    output_tensor = pixel_shuffle_cuda.pixel_unshuffle_nchw(input_tensor, (2, 2))
    output_tensor = pixel_shuffle_cuda.pixel_unshuffle_nchw(input_tensor, [2, 2])
    func = te.create_prim_func([input_tensor, output_tensor])
    ir_module_from_te = IRModule({"main": func})
    sch = tir.Schedule(ir_module_from_te)
    block_pixel_unshuffle = sch.get_block("PixelUnshuffle")
    (n, c, h, w) = sch.get_loops(block_pixel_unshuffle)
    no, ni = sch.split(n, factors=[None, 2])
    co, ci = sch.split(c, factors=[None, 32])
    ho, hi = sch.split(h, factors=[None, 2])
    wo, wi = sch.split(w, factors=[None, 2])
    sch.reorder(no, co, ho, wo, ni, ci, hi, wi)
    block_inner = sch.blockize(ni)

    a_shared = sch.cache_read(block_inner, 0, "shared")
    e_shared = sch.cache_write(block_inner, 0, "shared")
    sch.compute_at(a_shared, wo)
    sch.reverse_compute_at(e_shared, wo)
    ani, _, _, _ = sch.get_loops(a_shared)[-4:]
    sch.tensorize(ani, oraa_cuda.ORAA_LDG2S_N2C8H4W4_INT8_INTRIN)
    sch.tensorize(ni, oraa_cuda.ORAA_PIXEL_UNSHUFFLE_N2C8H4W4_INTRIN)
    e_n, _, _, _ = sch.get_loops(e_shared)[-4:]
    sch.tensorize(e_n, oraa_cuda.ORAA_STS2G_N2C32H2W2_INT8_INTRIN)
    sch.bind(no, "blockIdx.x")
    target = Target("oraa")
    # target = 'cuda'
    func = tvm.build(sch.mod, [input_tensor, output_tensor], target=target)
    print(func)
    print(func.imported_modules[0].get_source())


def relu(input_shape):
    a = te.placeholder(input_shape, dtype="int8", name="relu_a")
    b = te.compute(
        input_shape,
        lambda n, c, h, w: tir.max(a[n, c, h, w], tir.Cast("int8", 0)),
    )
    return [a, b]


def add(input_shape):
    a = te.placeholder(input_shape, dtype="int8", name="add_a")
    b = te.placeholder(input_shape, dtype="int8", name="add_b")
    c = te.placeholder(input_shape, dtype="int8", name="add_c")
    d = te.placeholder(input_shape, dtype="int8", name="add_d")
    e = te.compute(
        input_shape,
        lambda vn, vc, vh, vw: a[vn, vc, vh, vw]
        + b[vn, vc, vh, vw]
        + c[vn, vc, vh, vw]
        + d[vn, vc, vh, vw],
    )
    return [a, b, c, d, e]


def test_tensorize_oraa_add(input_shape):
    a = te.placeholder(input_shape, dtype="int8", name="add_a")
    b = te.placeholder(input_shape, dtype="int8", name="add_b")
    out = te.compute(
        input_shape,
        lambda vn, vc, vh, vw: a[vn, vc, vh, vw] + b[vn, vc, vh, vw],
    )
    func = te.create_prim_func([a, b, out])
    mod = IRModule({"main": func})
    sch = tir.Schedule(mod)
    # print(sch.mod)
    block = sch.get_block("compute")
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
    e_n, _, _, _ = sch.get_loops(e_shared)[-4:]
    sch.tensorize(e_n, oraa_cuda.ORAA_STS2G_N2C8H4W4_INT8_INTRIN)
    sch.tensorize(ni, oraa_cuda.ORAA_Add2_N2C8H4W4_INTRIN)
    sch.bind(no, "blockIdx.x")
    target = Target("oraa")
    # target = 'cuda'
    func = tvm.build(sch.mod, [a, b], target=target)
    print(func)
    print(func.imported_modules[0].get_source())
    a_data = tvm.nd.array(np.ones(input_shape,dtype="int8"),device=tvm.oraa())
    b_data = tvm.nd.array(np.ones(input_shape,dtype="int8"),device=tvm.oraa())
    c_data = tvm.nd.array(np.zeros(input_shape,dtype="int8"),device=tvm.oraa())
    func(a_data,b_data,c_data)


def test_tensorize_oraa_sub(input_shape):
    a = te.placeholder(input_shape, dtype="int8", name="add_a")
    b = te.placeholder(input_shape, dtype="int8", name="add_b")
    out = te.compute(
        input_shape,
        lambda vn, vc, vh, vw: a[vn, vc, vh, vw] - b[vn, vc, vh, vw],
    )
    func = te.create_prim_func([a, b, out])
    mod = IRModule({"main": func})
    sch = tir.Schedule(mod)
    # print(sch.mod)
    block = sch.get_block("compute")
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
    e_n, _, _, _ = sch.get_loops(e_shared)[-4:]
    sch.tensorize(e_n, oraa_cuda.ORAA_STS2G_N2C8H4W4_INT8_INTRIN)
    sch.tensorize(ni, oraa_cuda.ORAA_Sub_N2C8H4W4_INTRIN)
    sch.bind(no, "blockIdx.x")
    target = Target("oraa")
    # target = 'cuda'
    func = tvm.build(sch.mod, [a, b], target=target)
    print(func)
    print(func.imported_modules[0].get_source())


def test_tensorize_oraa_add4(input_shape):
    a, b, c, d, e = add(input_shape)
    func = te.create_prim_func([a, b, c, d, e])
    mod = IRModule({"main": func})
    sch = tir.Schedule(mod)
    # print(sch.mod)
    block = sch.get_block("compute")
    (n, c, h, w) = sch.get_loops(block)
    no, ni = sch.split(n, factors=[None, 2])
    co, ci = sch.split(c, factors=[None, 8])
    ho, hi = sch.split(h, factors=[None, 4])
    wo, wi = sch.split(w, factors=[None, 4])
    sch.reorder(no, co, ho, wo, ni, ci, hi, wi)
    block_inner = sch.blockize(ni)
    a_shared = sch.cache_read(block_inner, 0, "shared")
    b_shared = sch.cache_read(block_inner, 1, "shared")
    c_shared = sch.cache_read(block_inner, 2, "shared")
    d_shared = sch.cache_read(block_inner, 3, "shared")
    e_shared = sch.cache_write(block_inner, 0, "shared")
    sch.compute_at(a_shared, wo)
    sch.compute_at(b_shared, wo)
    sch.compute_at(c_shared, wo)
    sch.compute_at(d_shared, wo)
    sch.reverse_compute_at(e_shared, wo)
    # print(sch.mod)
    ani, _, _, _ = sch.get_loops(a_shared)[-4:]
    sch.tensorize(ani, oraa_cuda.ORAA_LDG2S_N2C8H4W4_INT8_INTRIN)
    bni, _, _, _ = sch.get_loops(b_shared)[-4:]
    sch.tensorize(bni, oraa_cuda.ORAA_LDG2S_N2C8H4W4_INT8_INTRIN)
    cni, _, _, _ = sch.get_loops(c_shared)[-4:]
    sch.tensorize(cni, oraa_cuda.ORAA_LDG2S_N2C8H4W4_INT8_INTRIN)
    dni, _, _, _ = sch.get_loops(d_shared)[-4:]
    sch.tensorize(dni, oraa_cuda.ORAA_LDG2S_N2C8H4W4_INT8_INTRIN)
    e_n, _, _, _ = sch.get_loops(e_shared)[-4:]
    sch.tensorize(e_n, oraa_cuda.ORAA_STS2G_N2C8H4W4_INT8_INTRIN)
    sch.tensorize(ni, oraa_cuda.ORAA_ADD4_N2C8H4W4_INTRIN)
    sch.bind(no, "blockIdx.x")
    target = Target("oraa")
    # target = 'cuda'
    func = tvm.build(sch.mod, [a, b], target=target)
    print(func)
    print(func.imported_modules[0].get_source())


def test_tensorize_oraa_relu(input_shape):
    a, b = relu(input_shape)
    func = te.create_prim_func([a, b])
    mod = IRModule({"main": func})
    sch = tir.Schedule(mod)
    block = sch.get_block("compute")
    # fused_axis = sch.fuse(*sch.get_loops(block))
    # fused_out, tile_n, tile_c, tile_w, tile_h = sch.split(fused_axis, factors=[None, 2, 2, 8, 2])
    (n, c, h, w) = sch.get_loops(block)
    no, ni = sch.split(n, factors=[None, 2])
    co, ci = sch.split(c, factors=[None, 2])
    ho, hi = sch.split(h, factors=[None, 8])
    wo, wi = sch.split(w, factors=[None, 2])
    sch.reorder(no, co, ho, wo, ni, ci, hi, wi)
    block_inner = sch.blockize(ni)

    a_shared = sch.cache_read(block_inner, 0, "shared")
    b_shared = sch.cache_write(block_inner, 0, "shared")
    sch.compute_at(a_shared, wo)
    sch.reverse_compute_at(b_shared, wo)
    ani, _, _, _ = sch.get_loops(a_shared)[-4:]
    sch.tensorize(ani, oraa_cuda.ORAA_LDG2S_N2C2H8W2_INT8_INTRIN)
    b_n, _, _, _ = sch.get_loops(b_shared)[-4:]
    sch.tensorize(b_n, oraa_cuda.ORAA_STS2G_N2C2H8W2_INT8_INTRIN)
    sch.tensorize(ni, oraa_cuda.ORAA_RELU_N2C2H8W2_INTRIN)
    sch.bind(no, "blockIdx.x")
    # sch.bind(co, "blockIdx.y")
    # sch.bind(ho, "threadIdx.x")
    # sch.bind(wo, "threadIdx.y")
    target = Target("oraa")
    # target = 'cuda'
    func = tvm.build(sch.mod, [a, b], target=target)
    print(func)
    print(func.imported_modules[0].get_source())
    input_np = np.array(np.random.randn(*input_shape) * 128, dtype=np.byte)
    output_np = np.array(np.zeros(input_shape), dtype=np.byte)
    input_tvm = tvm.nd.array(input_np, device=tvm.cuda(0))
    output_tvm = tvm.nd.array(output_np, device=tvm.cuda(0))


if __name__ == "__main__":
    # test_tensorize_oraa_relu((2, 16, 8, 8))
    # test_tensorize_oraa_relu((256, 256, 64, 64))
    # test_tensorize_oraa_pixel_shuffle((2, 16, 8, 8))
    # test_tensorize_oraa_pixel_unshuffle((2, 8, 4, 4))
    # test_tensorize_oraa_add4((2, 16, 8, 8))
    test_tensorize_oraa_add((2, 16, 8, 8))
    # test_tensorize_oraa_sub((2, 16, 8, 8))
