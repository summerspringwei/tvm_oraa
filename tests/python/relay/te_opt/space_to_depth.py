import tvm
from tvm import relay
from tvm import te
from tvm import tir
from tvm.relay.dataflow_pattern import is_op, wildcard
from tvm.relay.testing import run_opt_pass
from tvm.relay.backend.contrib.oraa import legalize
from tvm.script import tir as T
from tvm.topi import transform
import numpy as np


def make_pixel_shuffle_pattern():
    r"""Create a pattern to match the following graph.

    reshape
       |
    transpose
       |
    reshape
    """

    x = wildcard()
    reshape_node = is_op("reshape")(x)
    transpose_node = is_op("transpose")(reshape_node).has_attr(
        {"axes": [0, 1, 4, 2, 5, 3]})
    r = is_op("reshape")(transpose_node)
    return r


def make_space_to_depth_pattern():
    r"""Create a pattern to match the following graph.

    reshape
       |
    transpose
       |
    reshape
    """

    x = wildcard()
    reshape_node = is_op("reshape")(x)
    transpose_node = is_op("transpose")(reshape_node).has_attr(
        {"axes": [0, 3, 5, 1, 2, 4]})
    r = is_op("reshape")(transpose_node)
    return r


def make_space_to_depth_function(
        shape=(1, 16, 56, 56), downscale_factor=2, dtype="int8"):
    r"""Create space_to_depth relay function with shape,downscale_factor,dtype.
    """

    x = relay.var("x", shape=shape, dtype=dtype)
    reshape_node = relay.reshape(x, [
        shape[0], shape[1], shape[2] // downscale_factor, downscale_factor,
        shape[3] // downscale_factor, downscale_factor
    ])
    transpose_node = relay.transpose(reshape_node, [0, 3, 5, 1, 2, 4])
    reshape_node_2 = relay.reshape(transpose_node, [
        shape[0], shape[1] * downscale_factor * downscale_factor,
        shape[2] / downscale_factor, shape[3] / downscale_factor
    ])
    return relay.Function([x], reshape_node_2)


PATTERN_TABLE = [
    ("pixel_shuffle", make_pixel_shuffle_pattern()),
    ("space_to_depth", make_space_to_depth_pattern()),
]


def test():
    # relay Ops
    x = relay.var("x", shape=(1, 3, 224, 224))
    f = make_space_to_depth_function(shape=(1, 3, 224, 224))
    print(f, "\n", "*" * 10)
    annotated = run_opt_pass(f,
                             relay.transform.MergeComposite(PATTERN_TABLE),
                             import_prelude=False)
    print(annotated, "\n", "*" * 10)
    rewrited = legalize.transform_oraa_function(annotated)
    print(rewrited, "\n", "*" * 10)
    mod = tvm.IRModule.from_expr(rewrited)
    mod = relay.transform.InferType()(mod)
    print(mod, "\n", "*" * 10)

    # tensor expression to do space_to_depth compute
    N = te.var("N")
    C = te.var("C")
    H = te.var("H")
    W = te.var("W")
    Input = te.placeholder((N, C, H, W), name="Input",dtype="int8")
    #TODO: fix the correct te expression to do below:
    # Input[b][c][i][j] --> Output[b][(2*i%2+j%2)*C+c][i/2][j/2]
    Output = te.compute(
        (N, C * 4, H // 2, W // 2),
        lambda b, c, i, j: Input[b,
                                 (2 * (i % 2) + (j % 2)) * C + c, i // 2, j // 2],
        name="Output")
    s = te.create_schedule(Output.op)
    print(tvm.lower(s, [Input, Output], simple_mode=True))
    func = tvm.build(s, [Input, Output], "llvm")

    x = np.arange(96, dtype="int8")
    x = np.reshape(x, (1, 2, 6, 8))

    x = tvm.nd.array(x)
    print(x)
    y = tvm.nd.array(np.zeros((1, 8, 3, 4), dtype="int8"))
    print(y)
    func(x, y)
    print(y)


if __name__ == "__main__":
    test()
