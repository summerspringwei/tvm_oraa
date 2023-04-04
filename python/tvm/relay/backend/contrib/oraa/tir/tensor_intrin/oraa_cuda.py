"""Intrinsics for Open Research AI Architecture
"""
# pylint: disable=invalid-name
import tvm
from tvm.script import tir as T
from tvm.tir import TensorIntrin


@T.prim_func
def pixel_shuffle_n2c8h4w4_desc(a: T.handle, b: T.handle) -> None:
    """
    Int8 pixel shuffle description by shape (2, 8, 4, 4) with NCHW layout

    Parameters
    ----------
    a: tvm.script.tir.handle
      the input data pointer
    b: tvm.script.tir.handle
      the output data pointer
    """
    A = T.match_buffer(a, (2, 8, 4, 4), dtype="int8")
    B = T.match_buffer(b, (2, 2, 8, 8), dtype="int8")
    with T.block("root"):
        T.reads(A[0:2, 0:8, 0:4, 0:4])
        T.writes(B[0:2, 0:2, 0:8, 0:8])
        for n, c, h, w in T.grid(2, 2, 8, 8):
            with T.block("pixel_shuffel_basic_block"):
                vn, vc, vh, vw = T.axis.remap("SSSS", [n, c, h, w])
                B[vn, vc, vh, vw] = A[
                    vn,
                    vc * 2 * 2 + tvm.tir.indexmod(vh, 2) * 2 + tvm.tir.indexmod(vw, 2),
                    tvm.tir.indexdiv(vh, 2),
                    tvm.tir.indexdiv(vw, 2),
                ]


@T.prim_func
def pixel_shuffle_n2c8h4w4_impl(a: T.handle, b: T.handle) -> None:
    """
    Int8 pixel shuffle implementation by shape (2, 8, 4, 4) with NCHW layout
    Note: This tensorize implemetation is just a demonstration and not semantic equal!

    Parameters
    ----------
    a: tvm.script.tir.handle
      the input data pointer
    b: tvm.script.tir.handle
      the output data pointer
    """
    A = T.match_buffer(a, (2, 8, 4, 4), dtype="int8")
    B = T.match_buffer(b, (2, 2, 8, 8), dtype="int8")
    with T.block("root"):
        T.reads(A[0:2, 0:8, 0:4, 0:4])
        T.writes(B[0:2, 0:2, 0:8, 0:8])
        T.evaluate(
            T.call_extern(
                "vec4add",
                A.data,
                B.elem_offset,
                A.data,
                A.elem_offset,
                B.data,
                B.elem_offset,
                dtype="int8",
            )
        )


ORAA_PIXEL_SHUFFLE_N2C8H4W4_INTRIN = "oraa_pixel_shuffle_n2c8h4w4"

TensorIntrin.register(
    ORAA_PIXEL_SHUFFLE_N2C8H4W4_INTRIN, pixel_shuffle_n2c8h4w4_desc, pixel_shuffle_n2c8h4w4_impl
)


@T.prim_func
def pixel_unshuffle_n2c8h4w4_desc(a: T.handle, b: T.handle) -> None:
    """
    Int8 pixel unshuffle description by shape (2, 8, 4, 4) with NCHW layout

    Parameters
    ----------
    a: tvm.script.tir.handle
      the input data pointer
    b: tvm.script.tir.handle
      the output data pointer
    """

    A = T.match_buffer(a, (2, 8, 4, 4), dtype="int8")
    B = T.match_buffer(b, (2, 32, 2, 2), dtype="int8")
    with T.block("root"):
        T.reads(A[0:2, 0:8, 0:4, 0:4])
        T.writes(B[0:2, 0:32, 0:2, 0:2])
        for n, c, h, w in T.grid(2, 32, 2, 2):
            with T.block("pixel_unshuffle_basic_block"):
                vn, vc, vh, vw = T.axis.remap("SSSS", [n, c, h, w])
                B[vn, vc, vh, vw] = A[
                    vn,
                    tvm.tir.indexmod(vc, 8),
                    tvm.tir.indexdiv(tvm.tir.indexdiv(vc, 8), 2) + (vh * 2),
                    tvm.tir.indexmod(tvm.tir.indexdiv(vc, 8), 2) + (vw * 2),
                ]


@T.prim_func
def pixel_unshuffle_n2c8h4w4_impl(a: T.handle, b: T.handle) -> None:
    """
    Int8 pixel unshuffle implementation by shape (2, 8, 4, 4) with NCHW layout
    Note: This tensorize implemetation is just a demonstration and not semantic equal!

    Parameters
    ----------
    a: tvm.script.tir.handle
      the input data pointer
    b: tvm.script.tir.handle
      the output data pointer
    """
    A = T.match_buffer(a, (2, 8, 4, 4), dtype="int8")
    B = T.match_buffer(b, (2, 32, 2, 2), dtype="int8")
    with T.block("root"):
        T.reads(A[0:2, 0:8, 0:4, 0:4])
        T.writes(B[0:2, 0:32, 0:2, 0:2])
        T.evaluate(
            T.call_extern(
                "vec4add",
                A.data,
                B.elem_offset,
                A.data,
                A.elem_offset,
                B.data,
                B.elem_offset,
                dtype="int8",
            )
        )


ORAA_PIXEL_UNSHUFFLE_N2C8H4W4_INTRIN = "oraa_pixel_unshuffle_n2c8h4w4"

TensorIntrin.register(
    ORAA_PIXEL_UNSHUFFLE_N2C8H4W4_INTRIN,
    pixel_unshuffle_n2c8h4w4_desc,
    pixel_unshuffle_n2c8h4w4_impl,
)


@T.prim_func
def add4_n2c8h4w4_desc(a: T.handle, b: T.handle, c: T.handle, d: T.handle, out: T.handle) -> None:
    """
    Int8 add4 description by shape (2, 8, 4, 4) with NCHW layout

    Parameters
    ----------
    a: tvm.script.tir.handle
      the input data pointer
    b: tvm.script.tir.handle
      the input data pointer
    c: tvm.script.tir.handle
      the input data pointer
    d: tvm.script.tir.handle
      the input data pointer
    out: tvm.script.tir.handle
      the output data pointer
    """

    A = T.match_buffer(a, (2, 8, 4, 4), dtype="int8")
    B = T.match_buffer(b, (2, 8, 4, 4), dtype="int8")
    C = T.match_buffer(c, (2, 8, 4, 4), dtype="int8")
    D = T.match_buffer(d, (2, 8, 4, 4), dtype="int8")
    Out = T.match_buffer(out, (2, 8, 4, 4), dtype="int8")
    with T.block("root"):
        T.reads(
            A[0:2, 0:8, 0:4, 0:4],
            B[0:2, 0:8, 0:4, 0:4],
            C[0:2, 0:8, 0:4, 0:4],
            D[0:2, 0:8, 0:4, 0:4],
        )
        T.writes(Out[0:2, 0:8, 0:4, 0:4])
        for n, c, h, w in T.grid(2, 8, 4, 4):
            with T.block("add4_basic_block"):
                vn, vc, vh, vw = T.axis.remap("SSSS", [n, c, h, w])
                Out[vn, vc, vh, vw] = (
                    A[vn, vc, vh, vw] + B[vn, vc, vh, vw] + C[vn, vc, vh, vw] + D[vn, vc, vh, vw]
                )


@T.prim_func
def add4_n2c8h4w4_impl(a: T.handle, b: T.handle, c: T.handle, d: T.handle, out: T.handle) -> None:
    """
    Int8 add4 implementation by shape (2, 8, 4, 4) with NCHW layout

    Parameters
    ----------
    a: tvm.script.tir.handle
      the input data pointer
    b: tvm.script.tir.handle
      the input data pointer
    c: tvm.script.tir.handle
      the input data pointer
    d: tvm.script.tir.handle
      the input data pointer
    out: tvm.script.tir.handle
      the output data pointer
    """
    A = T.match_buffer(a, (2, 8, 4, 4), dtype="int8")
    B = T.match_buffer(b, (2, 8, 4, 4), dtype="int8")
    C = T.match_buffer(c, (2, 8, 4, 4), dtype="int8")
    D = T.match_buffer(d, (2, 8, 4, 4), dtype="int8")
    Out = T.match_buffer(out, (2, 8, 4, 4), dtype="int8")
    with T.block("root"):
        T.reads(
            A[0:2, 0:8, 0:4, 0:4],
            B[0:2, 0:8, 0:4, 0:4],
            C[0:2, 0:8, 0:4, 0:4],
            D[0:2, 0:8, 0:4, 0:4],
        )
        T.writes(Out[0:2, 0:8, 0:4, 0:4])
        T.evaluate(
            T.call_extern(
                "vec4add",
                A.data,
                B.elem_offset,
                A.data,
                A.elem_offset,
                B.data,
                B.elem_offset,
                dtype="int8",
            )
        )


ORAA_ADD4_N2C8H4W4_INTRIN = "oraa_add4_n2c8h4w4"

TensorIntrin.register(
    ORAA_ADD4_N2C8H4W4_INTRIN,
    add4_n2c8h4w4_desc,
    add4_n2c8h4w4_impl,
)


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
