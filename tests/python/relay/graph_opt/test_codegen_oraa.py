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
    fused_outer = sch.fuse(no, co, ho, wo)
    sch.bind(fused_outer, "blockIdx.x")
    # shared_read_block_pixel_shuffle = sch.cache_read(block_pixel_shuffle, 0, "shared")
    # sch.compute_at(shared_read_block_pixel_shuffle, fused_outer)
    # shared_write_block_pixel_shuffle = sch.cache_write(block_pixel_shuffle, 0, "shared")
    # sch.reverse_compute_at(shared_write_block_pixel_shuffle, fused_outer)
    def split_read(block, idx, tile_shape):
        block_read = sch.cache_read(block, idx, "shared")
        fused = sch.fuse(*sch.get_loops(block_read))
        splited_axis = sch.split(fused, factors=(tile_shape))
        return block_read, splited_axis[-4]

    block_read, tensorize_axis = split_read(block_pixel_shuffle, 0, [None, 2, 8, 4, 4])
    sch.compute_at(block_read, fused_outer)

    sch.tensorize(ni, oraa_cuda.ORAA_PIXEL_SHUFFLE_N2C8H4W4_INTRIN)
    # block_tensorized_pixel_shuffle = sch.get_block("PixelShuffle_o")

    return sch


def test_oraa_build(input_shape: list):
    sch = test_tensorize_oraa_pixel_shuffle(input_shape)
    print(sch.mod.script())
    input_placeholder = te.placeholder(input_shape, dtype="int8", name="input_tvm")
    output_shape = input_shape
    output_placeholder = te.placeholder(output_shape, dtype="int8", name="output_tvm")
    target = Target("oraa")
    func = tvm.build(sch.mod, [input_placeholder, output_placeholder], target=target)
    print(func)
    input_np = np.array(np.random.randn(*input_shape) * 128, dtype=np.byte)
    output_np = np.array(np.zeros(output_shape), dtype=np.byte)
    input_tvm = tvm.nd.array(input_np, device=tvm.cuda(0))
    output_tvm = tvm.nd.array(output_np, device=tvm.cuda(0))
    # func(input_tvm, output_tvm)


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
    test_tensorize_oraa_relu((2, 16, 8, 8))
    # test_tensorize_oraa_relu((256, 256, 64, 64))
    # test_tensorize_oraa_pixel_shuffle((2, 16, 8, 8))
    # test_oraa_build((2, 16, 8, 8))
    test_tensorize_oraa_add((2, 16, 8, 8))