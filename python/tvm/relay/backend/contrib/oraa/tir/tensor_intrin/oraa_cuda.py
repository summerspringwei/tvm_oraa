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
    A = T.match_buffer(a, (2, 2, 8, 2), dtype='int8', scope="shared")
    B = T.match_buffer(b, (2, 2, 8, 2), dtype='int8', scope="shared")
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
    A = T.match_buffer(a, (2, 2, 8, 2), dtype='int8', scope="shared")
    B = T.match_buffer(b, (2, 2, 8, 2), dtype='int8', scope="shared")
    with T.block("root"):
        T.reads(A[0:2, 0:2, 0:8, 0:2])
        T.writes(B[0:2, 0:2, 0:8, 0:2])
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
            shared_handle,
            shared_shape,
            dtype,
            offset_factor=1,
            scope="shared"
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
        s0 = T.var("int32")
        s1 = T.var("int32")
        s2 = T.var("int32")
        s3 = T.var("int32")
        global_buff = T.match_buffer(
            global_handle,
            shared_shape,
            dtype,
            offset_factor=1,
            scope="global",
            strides=[s0, s1, s2, s3]
        )
        shared_buff = T.match_buffer(
            shared_handle,
            shared_shape,
            dtype,
            offset_factor=1,
            scope="shared"
        )
        n_dim, c_dim, h_dim, w_dim = shared_shape
        with T.block("root"):
            T.reads(global_buff[0:n_dim, 0:c_dim, 0:h_dim, 0:w_dim])
            T.writes(shared_buff[0:n_dim, 0:c_dim, 0:h_dim, 0:w_dim])
            T.evaluate(
                T.oraa_slice_tensor(n_dim, c_dim, h_dim, w_dim,
                                  shared_buff.data,
                                  global_buff.access_ptr("r"),
                                  global_buff.elem_offset,
                                  s0, s1, s2,
                                  dtype=dtype)
                )
        # for n, c, h, w in T.grid(1, 1, 1, 1):
        #     with T.block("global_to_shared"):
        #         vn, vc, vh, vw = T.axis.remap("SSSS", [n, c, h, w])
        #         T.reads(global_buff[0:n_dim, 0:c_dim, 0:h_dim, 0:w_dim])
        #         T.writes(shared_buff[0:n_dim, 0:c_dim, 0:h_dim, 0:w_dim])
        #         T.evaluate(
        #             T.call_intrin("handle",
        #                           "tir.oraa_slice_tensor",
        #                           shared_buff.access_ptr('w'),
        #                           global_buff[0:n_dim, 0:c_dim, 0:h_dim, 0:w_dim],
        #                           dtype)
        #         )
        # for n, c, h in T.grid(n_dim, c_dim, h_dim):
        #     with T.block("global_to_shared"):
        #         vn, vc, vh = T.axis.remap("SSS", [n, c, h])
        #         T.reads(global_buff[vn, vc, vh, 0:w_dim])
        #         T.writes(shared_buff[vn, vc, vh, 0:w_dim])
        #         T.evaluate(
        #             T.call_intrin("handle",
        #                           "tir.oraa_slice_tensor",
        #                           shared_buff.access_ptr('w'),
        #                           global_buff[vn, vc, vh, 0],
        #                           dtype)
        #         )
        # for n, c, h, w in T.grid(n_dim, c_dim, h_dim, w_dim):
        #     with T.block("global_to_shared"):
        #         vn, vc, vh, vw = T.axis.remap("SSSS", [n, c, h, w])
        #         T.reads(global_buff[vn, vc, vh, vw])
        #         T.writes(shared_buff[vn, vc, vh, vw])
        #         shared_buff[vn, vc, vh, vw] = global_buff[vn, vc, vh, vw]
        
            
    return ldtensor_desc, ldtensor_impl


def get_sttensor_intrin(shared_shape, dtype):
    @T.prim_func
    def sttensor_desc(shared_handle: T.handle, global_handle: T.handle):
        global_buff = T.match_buffer(
            global_handle,
            shared_shape,
            dtype,
            offset_factor=1,
            scope="global"
        )
        shared_buff = T.match_buffer(
            shared_handle,
            shared_shape,
            dtype,
            offset_factor=1,
            scope="shared"
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
        shared_buff = T.match_buffer(
            shared_handle,
            shared_shape,
            dtype,
            scope="shared"
        )
        n_dim, c_dim, h_dim, w_dim = shared_shape
        with T.block("root"):
            T.reads(shared_buff[0:n_dim, 0:c_dim, 0:h_dim, 0:w_dim])
            T.writes(global_buff[0:n_dim, 0:c_dim, 0:h_dim, 0:w_dim])
            
            T.evaluate(T.call_extern(
                "vec4add",
                shared_buff.data,
                shared_buff.elem_offset,
                shared_buff.data,
                shared_buff.elem_offset,
                global_buff.data,
                global_buff.elem_offset,
                dtype=dtype,
            ))
    
    return sttensor_desc, sttensor_impl


ORAA_RELU_N2C2H8W2_INTRIN = "oraa_relu_n2c2h8w2"

TensorIntrin.register(ORAA_RELU_N2C2H8W2_INTRIN,
                      relu_n2c2h8w2_desc, relu_n2c2h8w2_impl)

ORAA_PIXEL_SHUFFLE_N2C8H4W4_INTRIN = "oraa_pixel_shuffle_n2c8h4w4"

TensorIntrin.register(ORAA_PIXEL_SHUFFLE_N2C8H4W4_INTRIN,
                      pixel_shuffle_n2c8h4w4_desc, pixel_shuffle_n2c8h4w4_impl)

ORAA_LDG2S_N2C2H8W2_INT8_INTRIN = "oraa_ldg2s_n2c2h8w2_int8"
TensorIntrin.register(ORAA_LDG2S_N2C2H8W2_INT8_INTRIN,
                      *get_ldtensor_intrin([2, 2, 8, 2], "int8"))


ORAA_STS2G_N2C2H8W2_INT8_INTRIN = "oraa_sts2g_n2c2h8w2_int8"
TensorIntrin.register(ORAA_STS2G_N2C2H8W2_INT8_INTRIN,
                      *get_sttensor_intrin([2, 2, 8, 2], "int8"))
