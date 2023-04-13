"""Intrinsics for Open Research AI Architecture
"""
# pylint: disable=invalid-name
from numpy import multiply
import tvm
from tvm.script import tir as T
from tvm.tir import TensorIntrin


@T.prim_func
def relu_n2c2h8w2_desc(a: T.handle, b: T.handle) -> None:
    """
    Int8 add description by shape (2, 2, 8, 2) with NCHW layout

    Parameters
    ----------
    a: tvm.script.tir.handle
      the input data pointer
    b: tvm.script.tir.handle
      the output data pointer
    """
    A = T.match_buffer(a, (2, 2, 8, 2), dtype="int8", scope="shared")
    B = T.match_buffer(b, (2, 2, 8, 2), dtype="int8", scope="shared")
    with T.block("root"):
        T.reads(A[0:2, 0:2, 0:8, 0:2])
        T.writes(B[0:2, 0:2, 0:8, 0:2])
        for n, c, h, w in T.grid(2, 2, 8, 2):
            with T.block("pixel_shuffel_basic_block"):
                vn, vc, vh, vw = T.axis.remap("SSSS", [n, c, h, w])
                B[vn, vc, vh, vw] = tvm.tir.max(A[vn, vc, vh, vw], T.int8(0))


@T.prim_func
def relu_n2c2h8w2_impl(a: T.handle, b: T.handle) -> None:
    """
    Int8 pixel shuffle implementation by shape (2, 2, 8, 2) with NCHW layout
    Note: This tensorize implemetation is just a demonstration and not semantic equal!

    Parameters
    ----------
    a: tvm.script.tir.handle
      the input data pointer
    b: tvm.script.tir.handle
      the output data pointer
    """
    A = T.match_buffer(a, (2, 2, 8, 2), dtype="int8", scope="shared")
    B = T.match_buffer(b, (2, 2, 8, 2), dtype="int8", scope="shared")
    with T.block("root"):
        T.reads(A[0:2, 0:2, 0:8, 0:2])
        T.writes(B[0:2, 0:2, 0:8, 0:2])
        T.evaluate(
            T.call_extern(
                "relu",
                A.data,
                A.elem_offset,
                B.data,
                B.elem_offset,
                dtype="int8",
            )
        )


ORAA_RELU_N2C2H8W2_INTRIN = "oraa_relu_n2c2h8w2"

TensorIntrin.register(ORAA_RELU_N2C2H8W2_INTRIN, relu_n2c2h8w2_desc, relu_n2c2h8w2_impl)


@T.prim_func
def pointwise_conv2d_n2c8h2w8_oc8ic8kh1kw1_desc(
    input: T.handle, weight: T.handle, output: T.handle
) -> None:
    """
    Int8 conv2d description by shape (2, 8, 2, 8) with NCHW layout

    Parameters
    ----------
    input: tvm.script.tir.handle
      the input data pointer
    weight: tvm.script.tir.handle
      the weight data pointer
    output: tvm.script.tir.handle
      the output data pointer
    """
    A = T.match_buffer(input, (2, 8, 2, 8), dtype="int8")
    B = T.match_buffer(weight, (8, 8, 1, 1), dtype="int8")
    C = T.match_buffer(output, (2, 8, 2, 8), dtype="int8")
    # C[n][oc][h][w] = ∑(ic) A[n][ic][h][w]*B[oc][ic][0][0]
    with T.block("root"):
        with T.init():
            for n, oc, h, w in T.grid(2, 8, 2, 8):
                C[n, oc, h, w] = 0
        for n, oc, h, w, ic in T.grid(2, 8, 2, 8, 8):
            with T.block("PointwiseConv2dNCHW"):
                vn, voc, vh, vw = T.axis.remap("SSSS", [n, oc, h, w])
                vic = T.axis.remap("R", [ic])
                C[vn, voc, vh, vw] = C[vn, voc, vh, vw] + (A[vn, voc, vh, vw] * B[voc, vic, 0, 0])


@T.prim_func
def pointwise_conv2d_n2c8h2w8_oc8ic8kh1kw1_impl(
    input: T.handle, weight: T.handle, output: T.handle
) -> None:
    """
    Int8 conv2d implementation by shape (2, 8, 2, 8) with NCHW layout

    Parameters
    ----------
    input: tvm.script.tir.handle
      the input data pointer
    weight: tvm.script.tir.handle
      the weight data pointer
    output: tvm.script.tir.handle
      the output data pointer
    """
    A = T.match_buffer(input, (2, 8, 2, 8), dtype="int8")
    B = T.match_buffer(weight, (8, 8, 1, 1), dtype="int8")
    C = T.match_buffer(output, (2, 8, 2, 8), dtype="int8")

    # C[n][oc][h][w] = ∑(ic) A[n][ic][h][w]*B[oc][ic][0][0]
    with T.block("root"):
        T.reads(A[0:2, 0:8, 0:2, 0:8], B[0:8, 0:8, 0:1, 0:1], C[0:2, 0:8, 0:2, 0:8])
        T.writes(C[0:2, 0:8, 0:2, 0:8])
        T.evaluate(
            T.call_extern(
                "vec4add",
                A.data,
                B.elem_offset,
                B.data,
                B.elem_offset,
                C.data,
                C.elem_offset,
                dtype="int8",
            )
        )


ORAA_PWC_N2C8H2W8_INTRIN = "oraa_pointwise_conv2d_n2c8h2w8_oc8ic8kh1kw1"

TensorIntrin.register(
    ORAA_PWC_N2C8H2W8_INTRIN,
    pointwise_conv2d_n2c8h2w8_oc8ic8kh1kw1_desc,
    pointwise_conv2d_n2c8h2w8_oc8ic8kh1kw1_impl,
)


def get_binary_intrin(input_shape, op, dtype):

    if op == tvm.tir.add:
        opcode = "binary_add"
    elif op == tvm.tir.subtract:
        opcode = "binary_sub"
    elif op == tvm.tir.multiply:
        opcode = "binary_mul"
    elif op == tvm.tir.indexdiv:
        opcode = "binary_div"

    @T.prim_func
    def binary_desc(a: T.handle, b: T.handle, out: T.handle):
        n_dim, c_dim, h_dim, w_dim = input_shape
        A = T.match_buffer(a, input_shape, dtype=dtype, scope="shared")
        B = T.match_buffer(b, input_shape, dtype=dtype, scope="shared")
        Out = T.match_buffer(out, input_shape, dtype=dtype, scope="shared")
        with T.block("root"):
            T.reads(
                A[0:n_dim, 0:c_dim, 0:h_dim, 0:w_dim],
                B[0:n_dim, 0:c_dim, 0:h_dim, 0:w_dim],
            )
            T.writes(Out[0:n_dim, 0:c_dim, 0:h_dim, 0:w_dim])
            for n, c, h, w in T.grid(n_dim, c_dim, h_dim, w_dim):
                with T.block("add3_basic_block"):
                    vn, vc, vh, vw = T.axis.remap("SSSS", [n, c, h, w])
                    Out[vn, vc, vh, vw] = op(A[vn, vc, vh, vw], B[vn, vc, vh, vw])

    @T.prim_func
    def binary_impl(a: T.handle, b: T.handle, out: T.handle):
        n_dim, c_dim, h_dim, w_dim = input_shape
        A = T.match_buffer(a, input_shape, dtype=dtype, scope="shared")
        B = T.match_buffer(b, input_shape, dtype=dtype, scope="shared")
        Out = T.match_buffer(out, input_shape, dtype=dtype, scope="shared")
        with T.block("root"):
            T.reads(
                A[0:n_dim, 0:c_dim, 0:h_dim, 0:w_dim],
                B[0:n_dim, 0:c_dim, 0:h_dim, 0:w_dim],
            )
            T.writes(Out[0:n_dim, 0:c_dim, 0:h_dim, 0:w_dim])
            T.evaluate(
                T.call_extern(
                    opcode,
                    A.data,
                    A.elem_offset,
                    B.data,
                    B.elem_offset,
                    Out.data,
                    Out.elem_offset,
                    dtype=dtype,
                )
            )

    return binary_desc, binary_impl


def get_add3_intrin(input_shape, dtype):
    @T.prim_func
    def add3_desc(a: T.handle, b: T.handle, c: T.handle, out: T.handle):
        n_dim, c_dim, h_dim, w_dim = input_shape
        A = T.match_buffer(a, input_shape, dtype=dtype, scope="shared")
        B = T.match_buffer(b, input_shape, dtype=dtype, scope="shared")
        C = T.match_buffer(c, input_shape, dtype=dtype, scope="shared")
        Out = T.match_buffer(out, input_shape, dtype=dtype, scope="shared")
        with T.block("root"):
            T.reads(
                A[0:n_dim, 0:c_dim, 0:h_dim, 0:w_dim],
                B[0:n_dim, 0:c_dim, 0:h_dim, 0:w_dim],
                C[0:n_dim, 0:c_dim, 0:h_dim, 0:w_dim],
            )
            T.writes(Out[0:n_dim, 0:c_dim, 0:h_dim, 0:w_dim])
            for n, c, h, w in T.grid(n_dim, c_dim, h_dim, w_dim):
                with T.block("add3_basic_block"):
                    vn, vc, vh, vw = T.axis.remap("SSSS", [n, c, h, w])
                    Out[vn, vc, vh, vw] = A[vn, vc, vh, vw] + B[vn, vc, vh, vw] + C[vn, vc, vh, vw]

    @T.prim_func
    def add3_impl(a: T.handle, b: T.handle, c: T.handle, out: T.handle):
        n_dim, c_dim, h_dim, w_dim = input_shape
        A = T.match_buffer(a, input_shape, dtype=dtype, scope="shared")
        B = T.match_buffer(b, input_shape, dtype=dtype, scope="shared")
        C = T.match_buffer(c, input_shape, dtype=dtype, scope="shared")
        Out = T.match_buffer(out, input_shape, dtype=dtype, scope="shared")
        with T.block("root"):
            T.reads(
                A[0:n_dim, 0:c_dim, 0:h_dim, 0:w_dim],
                B[0:n_dim, 0:c_dim, 0:h_dim, 0:w_dim],
                C[0:n_dim, 0:c_dim, 0:h_dim, 0:w_dim],
            )
            T.writes(Out[0:n_dim, 0:c_dim, 0:h_dim, 0:w_dim])
            T.evaluate(
                T.call_extern(
                    "add3",
                    A.data,
                    A.elem_offset,
                    B.data,
                    B.elem_offset,
                    C.data,
                    C.elem_offset,
                    Out.data,
                    Out.elem_offset,
                    dtype=dtype,
                )
            )

    return add3_desc, add3_impl


def get_add4_intrin(input_shape, dtype):
    @T.prim_func
    def add4_desc(a: T.handle, b: T.handle, c: T.handle, d: T.handle, out: T.handle):
        n_dim, c_dim, h_dim, w_dim = input_shape
        A = T.match_buffer(a, input_shape, dtype=dtype, scope="shared")
        B = T.match_buffer(b, input_shape, dtype=dtype, scope="shared")
        C = T.match_buffer(c, input_shape, dtype=dtype, scope="shared")
        D = T.match_buffer(d, input_shape, dtype=dtype, scope="shared")
        Out = T.match_buffer(out, input_shape, dtype=dtype, scope="shared")
        with T.block("root"):
            T.reads(
                A[0:n_dim, 0:c_dim, 0:h_dim, 0:w_dim],
                B[0:n_dim, 0:c_dim, 0:h_dim, 0:w_dim],
                C[0:n_dim, 0:c_dim, 0:h_dim, 0:w_dim],
                D[0:n_dim, 0:c_dim, 0:h_dim, 0:w_dim],
            )
            T.writes(Out[0:n_dim, 0:c_dim, 0:h_dim, 0:w_dim])
            for n, c, h, w in T.grid(n_dim, c_dim, h_dim, w_dim):
                with T.block("add4_basic_block"):
                    vn, vc, vh, vw = T.axis.remap("SSSS", [n, c, h, w])
                    Out[vn, vc, vh, vw] = (
                        A[vn, vc, vh, vw]
                        + B[vn, vc, vh, vw]
                        + C[vn, vc, vh, vw]
                        + D[vn, vc, vh, vw]
                    )

    @T.prim_func
    def add4_impl(a: T.handle, b: T.handle, c: T.handle, d: T.handle, out: T.handle):
        n_dim, c_dim, h_dim, w_dim = input_shape
        A = T.match_buffer(a, input_shape, dtype=dtype, scope="shared")
        B = T.match_buffer(b, input_shape, dtype=dtype, scope="shared")
        C = T.match_buffer(c, input_shape, dtype=dtype, scope="shared")
        D = T.match_buffer(d, input_shape, dtype=dtype, scope="shared")
        Out = T.match_buffer(out, input_shape, dtype=dtype, scope="shared")
        with T.block("root"):
            T.reads(
                A[0:n_dim, 0:c_dim, 0:h_dim, 0:w_dim],
                B[0:n_dim, 0:c_dim, 0:h_dim, 0:w_dim],
                C[0:n_dim, 0:c_dim, 0:h_dim, 0:w_dim],
                D[0:n_dim, 0:c_dim, 0:h_dim, 0:w_dim],
            )
            T.writes(Out[0:n_dim, 0:c_dim, 0:h_dim, 0:w_dim])
            T.evaluate(
                T.call_extern(
                    "add4",
                    A.data,
                    A.elem_offset,
                    B.data,
                    B.elem_offset,
                    C.data,
                    C.elem_offset,
                    D.data,
                    D.elem_offset,
                    Out.data,
                    Out.elem_offset,
                    dtype=dtype,
                )
            )

    return add4_desc, add4_impl

# mod="DCR"
def get_pixelshuffle_intrin(input_shape, dtype):
    @T.prim_func
    def pixelshuffle_desc(a: T.handle, b: T.handle):
        A = T.match_buffer(a, input_shape, dtype, scope="shared")
        n_dim, c_dim, h_dim, w_dim = input_shape
        o_n_dim = n_dim
        o_c_dim = c_dim // 4
        o_h_dim = h_dim * 2
        o_w_dim = w_dim * 2
        output_shape = o_n_dim, o_c_dim, o_h_dim, o_w_dim
        B = T.match_buffer(b, output_shape, dtype, scope="shared")
        with T.block("root"):
            T.reads(A[0:n_dim, 0:c_dim, 0:h_dim, 0:w_dim])
            T.writes(B[0:o_n_dim, 0:o_c_dim, 0:o_h_dim, 0:o_w_dim])
            for n, c, h, w in T.grid(o_n_dim, o_c_dim, o_h_dim, o_w_dim):
                with T.block("pixel_shuffle_basic_block"):
                    vn, vc, vh, vw = T.axis.remap("SSSS", [n, c, h, w])
                    B[vn, vc, vh, vw] = A[
                        vn,
                        vc + tvm.tir.truncmod(vh,2) * 4 + tvm.tir.truncmod(vw,2) * 2,
                        tvm.tir.truncdiv(vh,2),
                        tvm.tir.truncdiv(vw,2),
                    ]

    @T.prim_func
    def pixelshuffle_impl(a: T.handle, b: T.handle):
        A = T.match_buffer(a, input_shape, dtype, scope="shared")
        n_dim, c_dim, h_dim, w_dim = input_shape
        o_n_dim = n_dim
        o_c_dim = c_dim // 4
        o_h_dim = h_dim * 2
        o_w_dim = w_dim * 2
        output_shape = o_n_dim, o_c_dim, o_h_dim, o_w_dim
        B = T.match_buffer(b, output_shape, dtype, scope="shared")
        with T.block("root"):
            T.reads(A[0:n_dim, 0:c_dim, 0:h_dim, 0:w_dim])
            T.writes(B[0:o_n_dim, 0:o_c_dim, 0:o_h_dim, 0:o_w_dim])
            T.evaluate(
                T.call_extern(
                    "pixel_shuffle",
                    # input
                    A.data,
                    A.elem_offset,
                    # output
                    B.data,
                    B.elem_offset,
                    # input_shape
                    n_dim,
                    c_dim,
                    h_dim,
                    w_dim,
                    # output_shape
                    o_n_dim,
                    o_c_dim,
                    o_h_dim,
                    o_w_dim,
                    dtype=dtype,
                )
            )

    return pixelshuffle_desc, pixelshuffle_impl


def get_pixelunshuffle_intrin(input_shape, dtype):
    @T.prim_func
    def pixelunshuffle_desc(a: T.handle, b: T.handle):
        A = T.match_buffer(a, input_shape, dtype, scope="shared")
        n_dim, c_dim, h_dim, w_dim = input_shape
        o_n_dim = n_dim
        o_c_dim = c_dim * 4
        o_h_dim = h_dim // 2
        o_w_dim = w_dim // 2
        output_shape = o_n_dim, o_c_dim, o_h_dim, o_w_dim
        B = T.match_buffer(b, output_shape, dtype, scope="shared")
        with T.block("root"):
            T.reads(A[0:n_dim, 0:c_dim, 0:h_dim, 0:w_dim])
            T.writes(B[0:o_n_dim, 0:o_c_dim, 0:o_h_dim, 0:o_w_dim])
            for n, c, h, w in T.grid(o_n_dim, o_c_dim, o_h_dim, o_w_dim):
                with T.block("pixel_unshuffle_basic_block"):
                    vn, vc, vh, vw = T.axis.remap("SSSS", [n, c, h, w])
                    B[vn, vc, vh, vw] = A[
                        vn,
                        tvm.tir.floormod(vc, c_dim),
                        tvm.tir.floordiv(tvm.tir.floordiv(vc, c_dim), 2) + (vh * 2),
                        tvm.tir.floormod(tvm.tir.floordiv(vc, c_dim), 2) + (vw * 2),
                    ]

    @T.prim_func
    def pixelunshuffle_impl(a: T.handle, b: T.handle):
        A = T.match_buffer(a, input_shape, dtype, scope="shared")
        n_dim, c_dim, h_dim, w_dim = input_shape
        o_n_dim = n_dim
        o_c_dim = c_dim * 4
        o_h_dim = h_dim // 2
        o_w_dim = w_dim // 2
        output_shape = o_n_dim, o_c_dim, o_h_dim, o_w_dim
        B = T.match_buffer(b, output_shape, dtype, scope="shared")
        with T.block("root"):
            T.reads(A[0:n_dim, 0:c_dim, 0:h_dim, 0:w_dim])
            T.writes(B[0:o_n_dim, 0:o_c_dim, 0:o_h_dim, 0:o_w_dim])
            T.evaluate(
                T.call_extern(
                    "pixel_unshuffle",
                    # input
                    A.data,
                    A.elem_offset,
                    # output
                    B.data,
                    B.elem_offset,
                    # input_shape
                    n_dim,
                    c_dim,
                    h_dim,
                    w_dim,
                    # output_shape
                    o_n_dim,
                    o_c_dim,
                    o_h_dim,
                    o_w_dim,
                    dtype=dtype,
                )
            )

    return pixelunshuffle_desc, pixelunshuffle_impl


# api.write_to_tensor
def get_ldtensor_intrin(shared_shape, dtype):
    @T.prim_func
    def ldtensor_desc(shared_handle: T.handle, global_handle: T.handle):
        global_buff = T.match_buffer(
            global_handle,
            shared_shape,
            dtype,
            offset_factor=1,
            scope="global",
        )
        shared_buff = T.match_buffer(
            shared_handle, shared_shape, dtype, offset_factor=1, scope="shared"
        )
        n_dim, c_dim, h_dim, w_dim = shared_shape
        with T.block("root"):
            T.reads(global_buff[0:n_dim, 0:c_dim, 0:h_dim, 0:w_dim])
            T.writes(shared_buff[0:n_dim, 0:c_dim, 0:h_dim, 0:w_dim])

            for n, c, h, w in T.grid(n_dim, c_dim, h_dim, w_dim):
                with T.block("global_to_shared"):
                    vn, vc, vh, vw = T.axis.remap("SSSS", [n, c, h, w])
                    T.reads(global_buff[vn, vc, vh, vw])
                    T.writes(shared_buff[vn, vc, vh, vw])
                    shared_buff[vn, vc, vh, vw] = global_buff[vn, vc, vh, vw]

    @T.prim_func
    def ldtensor_impl(shared_handle: T.handle, global_handle: T.handle):
        global_buff = T.match_buffer(
            global_handle, shared_shape, dtype, offset_factor=1, scope="global"
        )
        shared_buff = T.match_buffer(
            shared_handle, shared_shape, dtype, offset_factor=1, scope="shared"
        )
        n_dim, c_dim, h_dim, w_dim = shared_shape
        s0 = T.var("int32")
        s1 = T.var("int32")
        s2 = T.var("int32")
        s3 = T.var("int32")
        with T.block("root"):
            T.reads(global_buff[0:n_dim, 0:c_dim, 0:h_dim, 0:w_dim])
            T.writes(shared_buff[0:n_dim, 0:c_dim, 0:h_dim, 0:w_dim])
            T.evaluate(
                T.oraa_slice_tensor(
                    shared_buff.data,
                    global_buff.data,
                    global_buff.elem_offset,
                    s0,
                    s1,
                    s2,
                    s3,
                    n_dim,
                    c_dim,
                    h_dim,
                    w_dim,
                    dtype=dtype,
                )
            )
            T.evaluate(
                T.call_extern(
                    "write_to_tensor",
                    shared_buff.data,
                    shared_buff.elem_offset,
                    global_buff.data,
                    global_buff.elem_offset,
                    dtype=dtype,
                )
            )

    return ldtensor_desc, ldtensor_impl


# api.read_from_tensor
def get_sttensor_intrin(shared_shape, dtype):
    @T.prim_func
    def sttensor_desc(shared_handle: T.handle, global_handle: T.handle):
        global_buff = T.match_buffer(
            global_handle, shared_shape, dtype, offset_factor=1, scope="global"
        )
        shared_buff = T.match_buffer(
            shared_handle, shared_shape, dtype, offset_factor=1, scope="shared"
        )
        n_dim, c_dim, h_dim, w_dim = shared_shape
        with T.block("root"):
            T.reads(shared_buff[0:n_dim, 0:c_dim, 0:h_dim, 0:w_dim])
            T.writes(global_buff[0:n_dim, 0:c_dim, 0:h_dim, 0:w_dim])

            for n, c, h, w in T.grid(n_dim, c_dim, h_dim, w_dim):
                with T.block("shared_to_global"):
                    vn, vc, vh, vw = T.axis.remap("SSSS", [n, c, h, w])
                    T.reads(shared_buff[vn, vc, vh, vw])
                    T.writes(global_buff[vn, vc, vh, vw])
                    global_buff[vn, vc, vh, vw] = shared_buff[vn, vc, vh, vw]

    @T.prim_func
    def sttensor_impl(shared_handle: T.handle, global_handle: T.handle):
        global_buff = T.match_buffer(
            global_handle,
            shared_shape,
            dtype,
            scope="global",
            offset_factor=1,
        )
        shared_buff = T.match_buffer(shared_handle, shared_shape, dtype, scope="shared")
        n_dim, c_dim, h_dim, w_dim = shared_shape
        with T.block("root"):
            T.reads(shared_buff[0:n_dim, 0:c_dim, 0:h_dim, 0:w_dim])
            T.writes(global_buff[0:n_dim, 0:c_dim, 0:h_dim, 0:w_dim])
            T.evaluate(
                T.call_extern(
                    "read_from_tensor",
                    shared_buff.data,
                    shared_buff.elem_offset,
                    global_buff.data,
                    global_buff.elem_offset,
                    dtype=dtype,
                )
            )

    return sttensor_desc, sttensor_impl


##### ld/st intrinsic, should match lots of inner loops
# 2282 block
ORAA_LDG2S_N2C2H8W2_INT8_INTRIN = "oraa_ldg2s_n2c2h8w2_int8"
TensorIntrin.register(ORAA_LDG2S_N2C2H8W2_INT8_INTRIN, *get_ldtensor_intrin([2, 2, 8, 2], "int8"))

ORAA_STS2G_N2C2H8W2_INT8_INTRIN = "oraa_sts2g_n2c2h8w2_int8"
TensorIntrin.register(ORAA_STS2G_N2C2H8W2_INT8_INTRIN, *get_sttensor_intrin([2, 2, 8, 2], "int8"))

# 2844 block
ORAA_LDG2S_N2C8H4W4_INT8_INTRIN = "oraa_ldg2s_n2c8h4w4_int8"
TensorIntrin.register(ORAA_LDG2S_N2C8H4W4_INT8_INTRIN, *get_ldtensor_intrin([2, 8, 4, 4], "int8"))

ORAA_STS2G_N2C8H4W4_INT8_INTRIN = "oraa_sts2g_n2c8h4w4_int8"
TensorIntrin.register(ORAA_STS2G_N2C8H4W4_INT8_INTRIN, *get_sttensor_intrin([2, 8, 4, 4], "int8"))

# 2288 block
ORAA_LDG2S_N2C2H8W8_INT8_INTRIN = "oraa_ldg2s_n2c2h8w8_int8"
TensorIntrin.register(ORAA_LDG2S_N2C2H8W8_INT8_INTRIN, *get_ldtensor_intrin([2, 2, 8, 8], "int8"))

ORAA_STS2G_N2C2H8W8_INT8_INTRIN = "oraa_sts2g_n2c2h8w8_int8"
TensorIntrin.register(ORAA_STS2G_N2C2H8W8_INT8_INTRIN, *get_sttensor_intrin([2, 2, 8, 8], "int8"))

# 2-32-2-2 block
ORAA_LDG2S_N2C32H2W2_INT8_INTRIN = "oraa_ldg2s_n2c32h2w2_int8"
TensorIntrin.register(ORAA_LDG2S_N2C32H2W2_INT8_INTRIN, *get_ldtensor_intrin([2, 32, 2, 2], "int8"))

ORAA_STS2G_N2C32H2W2_INT8_INTRIN = "oraa_sts2g_n2c32h2w2_int8"
TensorIntrin.register(ORAA_STS2G_N2C32H2W2_INT8_INTRIN, *get_sttensor_intrin([2, 32, 2, 2], "int8"))

##### compute intrinsic, only need to match minimum compute unit

ORAA_PIXEL_SHUFFLE_N2C8H4W4_INTRIN = "oraa_pixel_shuffle_n2c8h4w4"
TensorIntrin.register(
    ORAA_PIXEL_SHUFFLE_N2C8H4W4_INTRIN, *get_pixelshuffle_intrin([2, 8, 4, 4], "int8")
)

ORAA_PIXEL_UNSHUFFLE_N2C8H4W4_INTRIN = "oraa_pixel_unshuffle_n2c8h4w4"
TensorIntrin.register(
    ORAA_PIXEL_UNSHUFFLE_N2C8H4W4_INTRIN, *get_pixelunshuffle_intrin([2, 8, 4, 4], "int8")
)

ORAA_ADD4_N2C8H4W4_INTRIN = "oraa_add4_n2c8h4w4"
TensorIntrin.register(ORAA_ADD4_N2C8H4W4_INTRIN, *get_add4_intrin([2, 8, 4, 4], "int8"))

ORAA_Add3_N2C8H4W4_INTRIN = "oraa_add3_n2c8h4w4"
TensorIntrin.register(ORAA_Add3_N2C8H4W4_INTRIN, *get_add3_intrin([2, 8, 4, 4], "int8"))

ORAA_Add2_N2C8H4W4_INTRIN = "oraa_add_n2c8h4w4"
TensorIntrin.register(
    ORAA_Add2_N2C8H4W4_INTRIN, *get_binary_intrin([2, 8, 4, 4], tvm.tir.add, "int8")
)

ORAA_Sub_N2C8H4W4_INTRIN = "oraa_sub_n2c8h4w4"
TensorIntrin.register(
    ORAA_Sub_N2C8H4W4_INTRIN, *get_binary_intrin([2, 8, 4, 4], tvm.tir.subtract, "int8")
)

ORAA_Mul_N2C8H4W4_INTRIN = "oraa_mul_n2c8h4w4"
TensorIntrin.register(
    ORAA_Mul_N2C8H4W4_INTRIN, *get_binary_intrin([2, 8, 4, 4], tvm.tir.multiply, "int8")
)
