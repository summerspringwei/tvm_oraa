import tvm
from tvm import relay
from tvm import te
from tvm import tir
import numpy as np
from tvm import topi

def make_topi_depth_to_space_te(dtype="int8"):
    N = te.size_var("N")
    C = te.size_var("C")
    H = te.size_var("H")
    W = te.size_var("W")
    x = te.placeholder((N, C, H, W), name="x", dtype=dtype)
    topi_depth2space =  topi.nn.depth_to_space(x,2,layout="NCHW",mode="DCR")
    s = te.create_schedule(topi_depth2space.op)
    # print("DepthToSpace:", tvm.lower(s, [x, topi_depth2space],simple_mode=True))
    DepthToSpace = tvm.build(s, [x, topi_depth2space])
    return DepthToSpace


def make_topi_space_to_depth_te(dtype="int8"):
    N = te.size_var("N")
    C = te.size_var("C")
    H = te.size_var("H")
    W = te.size_var("W")
    x = te.placeholder((N, C, H, W), name="x", dtype=dtype)
    topi_space2depth =  topi.nn.space_to_depth(x,2,layout="NCHW")
    s = te.create_schedule(topi_space2depth.op)
    # print("SpaceToDepth:", tvm.lower(s, [x, topi_space2depth],simple_mode=True))
    SpaceToDepth = tvm.build(s, [x, topi_space2depth])
    return SpaceToDepth  

def make_space_to_depth_te(dtype="int8"):
    N = te.size_var("N")
    C = te.size_var("C")
    H = te.size_var("H")
    W = te.size_var("W")
    Input = te.placeholder((N, C, H, W), name="Input", dtype=dtype)

    def _get_pixel(n, c, y, x):
        block_offset = tvm.tir.truncdiv(c, C)
        channel_idx = tvm.tir.truncmod(c, C)
        x_idx = tvm.tir.truncmod(block_offset, 2)
        y_idx = tvm.tir.truncdiv(block_offset, 2)
        output = Input(n, channel_idx, y_idx + (y * 2), x_idx + (x * 2))
        return output

    Output = te.compute(
            (N, C * 4, H // 2, W // 2),
            lambda n, c, y, x: _get_pixel(n, c, y, x),
            name="Output")
    s = te.create_schedule(Output.op)
    # print("SpaceToDepth:", tvm.lower(s, [Input, Output], simple_mode=True))
    SpaceToDepth = tvm.build(s, [Input, Output])
    return SpaceToDepth

def make_depth_to_space_te(dtype="int8",mode="DCR"):
    N = te.size_var("N")
    C = te.size_var("C")
    H = te.size_var("H")
    W = te.size_var("W")
    Input = te.placeholder((N, C, H, W), name="Input", dtype=dtype)

    in_n, in_c, in_h, in_w = N,C,H,W
    channel_factor = tvm.tir.truncdiv(in_c, 4)
    output_shape = [in_n, channel_factor, in_h * 2, in_w * 2]

    def _get_pixel(n, c, y, x):
        block_x = tvm.tir.truncdiv(x, 2)
        block_y = tvm.tir.truncdiv(y, 2)
        idx_x = tvm.tir.truncmod(x, 2)
        idx_y = tvm.tir.truncmod(y, 2)
        if mode == "DCR":
            channel_idx = channel_factor * ((2 * idx_y) + idx_x) + c
        else:
            channel_idx = (c * 4) + ((2 * idx_y) + idx_x)
        output = Input(n, channel_idx, block_y, block_x)
        return output

    Output = te.compute((N, C // 4, H * 2, W * 2),
                        lambda n, c, y, x: _get_pixel(n, c, y, x),
                        name="Output")
    s = te.create_schedule(Output.op)
    # print("DepthToSpace:", tvm.lower(s, [Input, Output], simple_mode=True))
    DepthToSpace = tvm.build(s, [Input, Output])
    return DepthToSpace


def test_oraa_depth2space2depth(shape=(1, 8, 3, 4), dtype="int8"):
    DepthToSpace = make_depth_to_space_te(dtype=dtype)
    SpaceToDepth = make_space_to_depth_te(dtype=dtype)

    x = np.arange(shape[0] * shape[1] * shape[2] * shape[3], dtype=dtype)
    x = np.reshape(x, shape)
    x = tvm.nd.array(x)
    y = tvm.nd.array(
        np.zeros((shape[0], shape[1] // 4, shape[2] * 2, shape[3] * 2),
                 dtype=dtype))
    DepthToSpace(x, y)
    z = tvm.nd.array(
        np.zeros((shape[0], shape[1], shape[2], shape[3]), dtype=dtype))
    SpaceToDepth(y, z)
    assert np.array_equal(z.asnumpy(), x.asnumpy())
    print("oraa test successful")

def test_oraa_space2depth2space(shape=(1, 2, 6, 8), dtype="int8"):
    DepthToSpace = make_depth_to_space_te(dtype=dtype)
    SpaceToDepth = make_space_to_depth_te(dtype=dtype)

    x = np.arange(shape[0] * shape[1] * shape[2] * shape[3], dtype=dtype)
    x = np.reshape(x, shape)
    x = tvm.nd.array(x)
    y = tvm.nd.array(
        np.zeros((shape[0], shape[1] * 4, shape[2] // 2, shape[3] // 2),
                 dtype=dtype))
    SpaceToDepth(x, y)
    z = tvm.nd.array(
        np.zeros((shape[0], shape[1], shape[2], shape[3]), dtype=dtype))
    DepthToSpace(y, z)
    assert np.array_equal(z.asnumpy(), x.asnumpy())
    print("oraa test successful")


def test_topi_depth2space2depth(shape=(1, 8, 3, 4), dtype="int8"):
    x = np.arange(shape[0]*shape[1]*shape[2]*shape[3], dtype=dtype)
    x = np.reshape(x, shape)
    x = tvm.nd.array(x)
    topi_depth2space = make_topi_depth_to_space_te(dtype=dtype)
    topi_space2depth = make_topi_space_to_depth_te(dtype=dtype)
    y = tvm.nd.array(
        np.zeros((shape[0], shape[1]//4, shape[2]*2, shape[3]*2), dtype=dtype))
    z = tvm.nd.array(
        np.zeros((shape[0], shape[1], shape[2], shape[3]), dtype=dtype))
    topi_depth2space(x,y)
    # print(y)
    topi_space2depth(y,z)
    # print(z)
    assert np.array_equal(z.asnumpy(), x.asnumpy())
    print("topi test successful")





if __name__ == "__main__":
    test_oraa_space2depth2space()
    test_oraa_depth2space2depth()
    test_topi_depth2space2depth()
