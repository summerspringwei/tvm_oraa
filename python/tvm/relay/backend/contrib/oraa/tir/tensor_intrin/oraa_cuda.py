"""Intrinsics for Open Research AI Architecture

It seems there is bug in matching loops with extent being one.

@T.prim_func
def pixel_shuffle_C4H4W4_desc(
    A: T.Buffer((4, 4, 4), "int8", offset_factor=1),
    B: T.Buffer((1, 8, 8), "int8", offset_factor=1)
) -> None:
  with T.block("root"):
    T.reads(A[0:4, 0:4, 0:4])
    T.writes(B[0:1, 0:8, 0:8])
    for c in T.serial(0, 1):
      for h in T.serial(0, 8):
        for w in T.serial(0, 8):
          with T.block("pixel_shuffel_basic_block"):
            vc, vh, vw  = T.axis.remap("SSS", [c, h, w])
            B[vc, vh, vw] = A[vc * 2 * 2 + tvm.tir.indexmod(h, 2) * 2 + tvm.tir.indexmod(w, 2),
                                  tvm.tir.indexdiv(h, 2),
                                  tvm.tir.indexdiv(w, 2)]

@T.prim_func
def pixel_shuffle_C4H4W4_desc(
    # A: T.Buffer((4, 4, 4), "int8", offset_factor=1),
    # B: T.Buffer((1, 8, 8), "int8", offset_factor=1)
    a: T.handle, b: T.handle
) -> None:
  A = T.match_buffer(a, (1, 4, 4, 4), dtype='int8')
  B = T.match_buffer(b, (1, 1, 8, 8), dtype='int8')
  with T.block("root"):
    T.reads(A[:, 0:4, 0:4, 0:4])
    T.writes(B[:, 0:1, 0:8, 0:8])
    for n, c, h ,w in T.grid(1, 1, 8, 8):
    # for h, w in T.grid(8, 8):
      with T.block("pixel_shuffel_basic_block"):
        vh, vw  = T.axis.remap("SS", [h, w])
        nn = T.var("int32")
        cc = T.var("int32")
        B[vn, vc, vh, vw] = A[vn, vc * 2 * 2 + tvm.tir.indexmod(h, 2) * 2 + tvm.tir.indexmod(w, 2),
                              tvm.tir.indexdiv(h, 2),
                              tvm.tir.indexdiv(w, 2)]


@T.prim_func
def pixel_shuffle_C4H4W4_impl(
    A: T.Buffer((4, 4, 4), "int8", offset_factor=1),
    B: T.Buffer((1, 8, 8), "int8", offset_factor=1)
) -> None:
    with T.block("root"):
      T.reads(A[0:4, 0:4, 0:4])
      T.writes(B[0:1, 0:8, 0:8])
      for c in T.serial(0, 1):
        for h in T.serial(0, 8):
          for w in T.serial(0, 8):
            with T.block("pixel_shuffel_basic_block"):
              vc, vh, vw  = T.axis.remap("SSS", [c, h, w])
              B[vc, vh, vw] = A[vc * 2 * 2 + tvm.tir.indexmod(h, 2) * 2 + tvm.tir.indexmod(w, 2),
                                    tvm.tir.indexdiv(h, 2),
                                    tvm.tir.indexdiv(w, 2)]

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
    A = T.match_buffer(a, (2, 8, 4, 4), dtype='int8')
    B = T.match_buffer(b, (2, 2, 8, 8), dtype='int8')
    with T.block("root"):
        T.reads(A[0:2, 0:8, 0:4, 0:4])
        T.writes(B[0:2, 0:2, 0:8, 0:8])
        for n, c, h, w in T.grid(2, 2, 8, 8):
            with T.block("pixel_shuffel_basic_block"):
                vn, vc, vh, vw = T.axis.remap("SSSS", [n, c, h, w])
                B[vn, vc, vh,
                  vw] = A[vn, vc * 2 * 2 + tvm.tir.indexmod(vh, 2) * 2 +
                          tvm.tir.indexmod(vw, 2),
                          tvm.tir.indexdiv(vh, 2),
                          tvm.tir.indexdiv(vw, 2)]


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
    A = T.match_buffer(a, (2, 8, 4, 4), dtype='int8', scope="shared")
    B = T.match_buffer(b, (2, 2, 8, 8), dtype='int8', scope="shared")
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
            ))


ORAA_PIXEL_SHUFFLE_N2C8H4W4_INTRIN = "oraa_pixel_shuffle_n2c8h4w4"

TensorIntrin.register(ORAA_PIXEL_SHUFFLE_N2C8H4W4_INTRIN,
                      pixel_shuffle_n2c8h4w4_desc, pixel_shuffle_n2c8h4w4_impl)
