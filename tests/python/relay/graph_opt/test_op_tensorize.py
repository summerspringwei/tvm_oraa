import tvm
from tvm import relay, te, tir, topi
from tvm import IRModule
from tvm import meta_schedule as ms
from tvm.target import Target
from tvm.topi.utils import get_const_tuple
from tvm.relay.backend.contrib.oraa import op
from tvm.relay.backend.contrib.oraa.topi import pixel_shuffle_cuda
from tvm.relay.backend.contrib.oraa.tir.tensor_intrin import oraa_cuda
from tvm.topi import utils


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
    print(sch.mod.script())
    sch.tensorize(ni, oraa_cuda.ORAA_PIXEL_SHUFFLE_N2C8H4W4_INTRIN)

    # print(sch.mod.script())


def test_tensorize_oraa_pixel_unshuffle(input_shape: list):
    """Test tensorize of pixel unshuffle computation"""
    input_tensor = te.placeholder(input_shape, dtype="int8", name="input_name")
    output_tensor = pixel_shuffle_cuda.pixel_unshuffle_nchw(input_tensor, (2, 2))
    output_tensor = pixel_shuffle_cuda.pixel_unshuffle_nchw(input_tensor, [2, 2])
    func = te.create_prim_func([input_tensor, output_tensor])
    ir_module_from_te = IRModule({"main": func})

    sch = tir.Schedule(ir_module_from_te)
    # print(sch.mod.script())
    block_pixel_unshuffle = sch.get_block("PixelUnshuffle")
    (n, c, h, w) = sch.get_loops(block_pixel_unshuffle)
    no, ni = sch.split(n, factors=[None, 2])
    co, ci = sch.split(c, factors=[None, 32])
    ho, hi = sch.split(h, factors=[None, 2])
    wo, wi = sch.split(w, factors=[None, 2])
    sch.reorder(no, co, ho, wo, ni, ci, hi, wi)
    print(sch.mod.script())

    # sch.tensorize(ni, oraa_cuda.ORAA_PIXEL_UNSHUFFLE_N2C8H4W4_INTRIN)

    # print(sch.mod.script())


def pointwise_conv2d_nchw(Input, Filter, out_dtype="int8"):
    batch, in_channel, in_height, in_width = utils.get_const_tuple(Input.shape)
    out_channel, _, k_height, k_width = utils.get_const_tuple(Filter.shape)
    out_height = in_height
    out_width = in_width
    out_height = in_height
    out_width = in_width
    # N OC H W IC (KH KW)
    ic = te.reduce_axis((0, in_channel), name="ic")
    Output = te.compute(
        (batch, out_channel, out_height, out_width),
        lambda b, c, h, w: te.sum(Input[b, c, h, w] * Filter[c, ic, 0, 0], axis=ic),
        name="PointwiseConv2dNCHW",
    )
    return Output


def test_tensorize_oraa_pointwise_conv2d(input_shape: list):
    input_tensor = te.placeholder(input_shape, dtype="int8", name="input_name")
    weight_tensor = te.placeholder(
        shape=(input_shape[1], input_shape[1], 1, 1), dtype="int8", name="weight_name"
    )
    output_tensor = pointwise_conv2d_nchw(input_tensor, weight_tensor)
    func = te.create_prim_func([input_tensor, weight_tensor, output_tensor])
    # print(func)
    # official_out = topi.nn.conv2d_nchw(Input=input_tensor,Filter=weight_tensor,stride=1,padding=0,dilation=1,
    #                                     out_dtype="int8")
    # official_func = te.create_prim_func([input_tensor, weight_tensor, official_out])
    # print(official_func)
    ir_module_from_te = IRModule({"main": func})
    sch = tir.Schedule(ir_module_from_te)
    block_pointwise_conv2d = sch.get_block("PointwiseConv2dNCHW")
    plist = sch.get_loops(block_pointwise_conv2d)
    (n, oc, h, w, ic) = plist
    no, ni = sch.split(n, factors=[None, 2])
    co, ci = sch.split(oc, factors=[None, 8])
    ho, hi = sch.split(h, factors=[None, 2])
    wo, wi = sch.split(w, factors=[None, 8])
    sch.reorder(no, co, ho, wo, ni, ci, hi, wi, ic)
    sch.tensorize(ni, oraa_cuda.ORAA_PWC_N2C8H2W8_INTRIN)
    # print(sch.mod.script())


def test_tensorize_oraa_add4(input_shape: list):
    a = te.placeholder(input_shape, dtype="int8", name="a")
    b = te.placeholder(input_shape, dtype="int8", name="b")
    c = te.placeholder(input_shape, dtype="int8", name="c")
    d = te.placeholder(input_shape, dtype="int8", name="d")
    batch, channel, height, width = utils.get_const_tuple(a.shape)
    out = te.compute(
        (batch, channel, height, width),
        lambda vn, vc, vh, vw: a[vn, vc, vh, vw]
        + b[vn, vc, vh, vw]
        + c[vn, vc, vh, vw]
        + d[vn, vc, vh, vw],
        name="Add4NCHW",
    )
    func = te.create_prim_func([a, b, c, d, out])
    # print(func)
    ir_module_from_te = IRModule({"main": func})
    sch = tir.Schedule(ir_module_from_te)
    block_add4 = sch.get_block("Add4NCHW")
    plist = sch.get_loops(block_add4)
    (n, oc, h, w) = plist
    no, ni = sch.split(n, factors=[None, 2])
    co, ci = sch.split(oc, factors=[None, 8])
    ho, hi = sch.split(h, factors=[None, 4])
    wo, wi = sch.split(w, factors=[None, 4])
    sch.reorder(no, co, ho, wo, ni, ci, hi, wi)
    sch.tensorize(ni, oraa_cuda.ORAA_ADD4_N2C8H4W4_INTRIN)
    # print(sch.mod.script())


if __name__ == "__main__":
    # test_tensorize_oraa_pixel_shuffle((16, 16, 16, 16))
    test_tensorize_oraa_pixel_unshuffle((4, 32, 8, 8))
    test_tensorize_oraa_pointwise_conv2d((4, 8, 2, 8))
    # test_tensorize_oraa_add4((2, 8, 4, 4))
    print("Success")
