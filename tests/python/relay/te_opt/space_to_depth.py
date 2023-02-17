import tvm
from tvm import relay
from tvm import te
from tvm import tir
import numpy as np

def make_space_to_depth_te(dtype="int8"):
    N = te.size_var("N")
    C = te.size_var("C")
    H = te.size_var("H")
    W = te.size_var("W")
    Input = te.placeholder((N, C, H, W), name="Input", dtype=dtype)
    Output = te.compute(
        (N, C * 4, H // 2, W // 2),
        lambda b, c, i, j: Input[b,
                                 tir.indexmod(c, C), (2 * i) + tir.indexdiv(
                                     c, 2 * C), (2 * j) + tir.indexdiv(tir.indexmod(c,2*C), C)],
        name="Output")
    s = te.create_schedule(Output.op)
    print("SpaceToDepth:",tvm.lower(s, [Input, Output], simple_mode=True))
    SpaceToDepth = tvm.build(s, [Input, Output])
    return SpaceToDepth

def make_depth_to_space_te(dtype="int8"):
    N = te.size_var("N")
    C = te.size_var("C")
    H = te.size_var("H")
    W = te.size_var("W")
    Input = te.placeholder((N, C, H, W), name="Input", dtype=dtype)
    Output = te.compute(
        (N, C // 4, H * 2, W * 2),
        lambda b, c, i, j: Input[b,(2*tir.indexmod(i,2)+tir.indexmod(j,2))*(C//4)+c,tir.indexdiv(i,2), tir.indexdiv(j,2)],
        name="Output")
    s = te.create_schedule(Output.op)
    print("DepthToSpace:",tvm.lower(s, [Input, Output], simple_mode=True))
    DepthToSpace = tvm.build(s, [Input, Output])
    return DepthToSpace

def test_space2depth2space(shape=(1,2,6,8),dtype="int8"):
    SpaceToDepth = make_space_to_depth_te(dtype=dtype)
    DepthToSpace = make_depth_to_space_te(dtype=dtype)

    x = np.arange(shape[0]*shape[1]*shape[2]*shape[3], dtype=dtype)
    x = np.reshape(x, shape)

    x = tvm.nd.array(x)
    print(x)
    y = tvm.nd.array(np.zeros((shape[0],4*shape[1],shape[2]//2,shape[3]//2), dtype=dtype))
    SpaceToDepth(x, y)
    print(y)
    z = tvm.nd.array(np.zeros((shape[0],shape[1],shape[2],shape[3]), dtype=dtype))
    DepthToSpace(y, z)
    print(z)
    assert np.array_equal(z.asnumpy(), x.asnumpy())
    print("success")

def test_depth2space2depth(shape=(1,8,3,4),dtype="int8"):
    SpaceToDepth = make_space_to_depth_te(dtype=dtype)
    DepthToSpace = make_depth_to_space_te(dtype=dtype)

    x = np.arange(shape[0]*shape[1]*shape[2]*shape[3], dtype=dtype)
    x = np.reshape(x, shape)

    x = tvm.nd.array(x)
    print(x)
    y = tvm.nd.array(np.zeros((shape[0],shape[1]//4,shape[2]*2,shape[3]*2), dtype=dtype))
    DepthToSpace(x, y)
    print(y)
    z = tvm.nd.array(np.zeros((shape[0],shape[1],shape[2],shape[3]), dtype=dtype))
    SpaceToDepth(y, z)
    print(z)
    assert np.array_equal(z.asnumpy(), x.asnumpy())
    print("success")

if __name__ == "__main__":
    test_space2depth2space()
    test_depth2space2depth()
    test_space2depth2space(shape=(3,16,64,64),dtype="int32")
    test_depth2space2depth(shape=(3,8,32,32),dtype="float32")
