"""Test graph optimization rules for Open-Research-AI-Compiler"""

import tvm
from tvm import relay
from tvm import te
from tvm import tir
from tvm.relay.dataflow_pattern import is_op, wildcard
from tvm.relay.testing import run_opt_pass
from tvm.relay.backend.contrib.oraa import legalize

def get_root_call(call, root_op_name):
    if not isinstance(call, relay.Call):
        return None
    if str(call.op) == root_op_name:
        return call
    return get_root_call(call.args[0], root_op_name)


def make_add_relu_pattern():
    r"""Create a pattern to match the following graph.

     add
      |
    relu
    """
    add_node = wildcard() + wildcard()
    r = is_op("nn.relu")(add_node)
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
    transpose_node = is_op("transpose")(reshape_node).has_attr({"axes":[0,3,5,1,2,4]})
    r = is_op("reshape")(transpose_node)
    return r

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
    transpose_node = is_op("transpose")(reshape_node).has_attr({"axes":[0,1,4,2,5,3]})
    r = is_op("reshape")(transpose_node)
    return r


PATTERN_TABLE = [
    ("pixel_shuffle", make_pixel_shuffle_pattern()),
    ("space_to_depth", make_space_to_depth_pattern()),
]


def test_pixel_shuffle():
    r"""Test composite function is correctly produced from a graph with single branch.

    We would expect the pattern `make_pixel_shuffle_pattern()`
    to be merged into a single op `pixel_shuffle`
    """
    def before():
        x = relay.var("x", shape=(1, 16, 56, 56), dtype="int8")
        reshape_node = relay.reshape(x, [1, 16 / 4, 2, 2, 56, 56])
        transpose_node = relay.transpose(reshape_node, [0, 1, 4, 2, 5, 3])
        reshape_node_2 = relay.reshape(transpose_node,
                                       [1, 16 / 4, 56 * 2, 56 * 2])

        return relay.Function([x], reshape_node_2)

    def expected():
        x = relay.var("x", shape=(1, 16, 56, 56), dtype="int8")
        reshape_node = relay.reshape(x, [1, 16 / 4, 2, 2, 56, 56])
        transpose_node = relay.transpose(reshape_node, [0, 1, 4, 2, 5, 3])
        reshape_node_2 = relay.reshape(transpose_node,
                                       [1, 16 / 4, 56 * 2, 56 * 2])
        reshape_transpose_reshape = relay.Function([
            x,
        ], reshape_node_2)
        reshape_transpose_reshape = reshape_transpose_reshape.with_attr(
            "Composite", "pixel_shuffle")
        reshape_transpose_reshape = reshape_transpose_reshape.with_attr(
            "PartitionedFromPattern", "reshape_transpose_reshape_")

        # merged function
        pa = relay.var("pa", shape=(1, 16, 56, 56), dtype="int8")
        r = relay.Call(reshape_transpose_reshape, [
            pa,
        ])
        return relay.Function([pa], r)

    graph = before()
    result = run_opt_pass(graph,
                          relay.transform.MergeComposite(PATTERN_TABLE),
                          import_prelude=False)
    print("test pixel_shuffle:")
    print(graph)
    print("=" * 20)
    print(result)
    print("=" * 20)
    expected_graph = expected()
    expected_graph = run_opt_pass(expected_graph, relay.transform.InferType())
    assert tvm.ir.structural_equal(
        result, expected_graph, map_free_vars=True
    ), "Graph mismatch: output vs. expected\n{0}\n=====\n{1}".format(
        str(result), str(expected_graph))

    # rewrite
    rewrited = legalize.transform_oraa_function(result)
    print(rewrited)
    print("=" * 20)

def test_space_to_depth():
    r"""Test composite function is correctly produced from a graph with single branch.

    We would expect the pattern `make_space_to_depth_pattern()`
    to be merged into a single op `space_to_depth`
    """
    def before():
        x = relay.var("x", shape=(1, 16, 56, 56), dtype="int8")
        reshape_node = relay.reshape(x, [1, 16, 28, 2, 28, 2])
        transpose_node = relay.transpose(reshape_node, [0, 3, 5, 1, 2, 4])
        reshape_node_2 = relay.reshape(transpose_node,
                                       [1, 16 * 4, 56 / 2, 56 / 2])

        return relay.Function([x], reshape_node_2)

    def expected():
        x = relay.var("x", shape=(1, 16, 56, 56), dtype="int8")
        reshape_node = relay.reshape(x, [1, 16, 28, 2, 28, 2])
        transpose_node = relay.transpose(reshape_node, [0, 3, 5, 1, 2, 4])
        reshape_node_2 = relay.reshape(transpose_node,
                                       [1, 16 * 4, 56 / 2, 56 / 2])
        reshape_transpose_reshape = relay.Function([
            x,
        ], reshape_node_2)
        reshape_transpose_reshape = reshape_transpose_reshape.with_attr(
            "Composite", "space_to_depth")
        reshape_transpose_reshape = reshape_transpose_reshape.with_attr(
            "PartitionedFromPattern", "reshape_transpose_reshape_")

        # merged function
        pa = relay.var("pa", shape=(1, 16, 56, 56), dtype="int8")
        r = relay.Call(reshape_transpose_reshape, [
            pa,
        ])
        return relay.Function([pa], r)

    graph = before()
    result = run_opt_pass(graph,
                          relay.transform.MergeComposite(PATTERN_TABLE),
                          import_prelude=False)
    print("test space_to_depth:")
    print(graph)
    print("=" * 20)
    print(result)
    print("=" * 20)
    expected_graph = expected()
    expected_graph = run_opt_pass(expected_graph, relay.transform.InferType())
    assert tvm.ir.structural_equal(
        result, expected_graph, map_free_vars=True
    ), "Graph mismatch: output vs. expected\n{0}\n=====\n{1}".format(
        str(result), str(expected_graph))

    # rewrite
    rewrited = legalize.transform_oraa_function(result)
    print(rewrited)
    print("=" * 20)


def te_pixel_shuffle_nchw():
    input = te.placeholder((1, 64, 56, 56), dtype='int8')
    output = te.compute(
        (1, 16, 112, 112), lambda n, c, h, w: input[n, c * 4 + tir.indexmod(
            h, 2) * 2 + tir.indexmod(w, 2),
                                                    tir.indexdiv(h, 2),
                                                    tir.indexdiv(w, 2)])
    return output


if __name__ == "__main__":
    test_pixel_shuffle()
    test_space_to_depth()
    print("success")
