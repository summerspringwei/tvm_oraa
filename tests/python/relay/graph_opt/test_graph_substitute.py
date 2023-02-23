"""Test graph optimization rules for Open-Research-AI-Compiler"""

import tvm
from tvm import relay
from tvm import te
from tvm import tir
from tvm.relay.dataflow_pattern import is_op, wildcard
from tvm.relay.testing import run_opt_pass
from tvm.relay.backend.contrib.oraa import legalize


def make_add_relu_pattern():
    r"""Create a pattern to match the following graph.

     add
      |
    relu
    """
    add_node = wildcard() + wildcard()
    r = is_op("nn.relu")(add_node)
    return r


def make_add3_pattern():
    r"""Create a pattern the match the following graph:

    add
     |
    add
     |
    add
    """
    x, b, c = wildcard(), wildcard(), wildcard()
    add_1 = is_op("add")(x, b)
    add_2 = is_op("add")(add_1, c)

    return add_2


def make_add4_pattern():
    r"""Create a pattern the match the following graph:

    add
     |
    add
     |
    add
    """
    x, b, c, d = wildcard(), wildcard(), wildcard(), wildcard()
    add_1 = is_op("add")(x, b)
    add_2 = is_op("add")(add_1, c)
    add_3 = is_op("add")(add_2, d)

    return add_3


def make_reshape_transpose_reshape_pattern():
    r"""Create a pattern to match the following graph.

    reshape
       |
    transpose
       |
    reshape
    """

    x = wildcard()
    # We only need to match input tensor rather than other attributes here
    reshape_node = is_op("reshape")(x)
    transpose_node = is_op("transpose")(reshape_node)
    r = is_op("reshape")(transpose_node)
    return r


PATTERN_TABLE = [
    ("pixel_shuffle", make_reshape_transpose_reshape_pattern()),
    ("add4", make_add4_pattern()),
    ("add3", make_add3_pattern()),
]


def test_reshape_transpose_reshape():
    r"""Test composite function is correctly produced from a graph with single branch.

    We would expect the pattern `make_reshape_transpose_reshape_pattern`
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


def test_addN():
    r"""Test composite function is correctly produced with multiple add
    We would expect the pattern `make_add3_pattern` and `make_add4_pattern`
    to be to be merged into a single op `add3` and `add4`, respectively.

        a  b
         \/
         add c
           \/                   a b c d
           add d                 \   /
             \/        ====>      add4 e f
             add e                  \  /
               \/                   add3
               add f
                 \/
                 add
    """

    def before():
        a = relay.var("a", shape=(1, 16, 56, 56))
        b = relay.var("b", shape=(1, 16, 56, 56))
        c = relay.var("c", shape=(1, 16, 56, 56))
        d = relay.var("d", shape=(1, 16, 56, 56))
        e = relay.var("e", shape=(1, 16, 56, 56))
        f = relay.var("f", shape=(1, 16, 56, 56))

        r = a + b + c + d + e + f

        return relay.Function([a, b, c, d, e, f], r)

    def expect():
        # Declare add4 function
        a = relay.var("a", shape=(1, 16, 56, 56))
        b = relay.var("b", shape=(1, 16, 56, 56))
        c = relay.var("c", shape=(1, 16, 56, 56))
        d = relay.var("d", shape=(1, 16, 56, 56))
        add_4 = a + b + c + d
        func_add_4 = relay.Function([a, b, c, d], add_4)
        func_add_4 = func_add_4.with_attr("Composite", "add4")
        func_add_4 = func_add_4.with_attr("PartitionedFromPattern", "add_add_add_")
        
        # Declare add3 function
        e = relay.var("e", shape=(1, 16, 56, 56))
        f = relay.var("f", shape=(1, 16, 56, 56))
        g = relay.var("g", shape=(1, 16, 56, 56))
        add_3 = e + f + g
        func_add_3 = relay.Function([e, f, g], add_3)
        func_add_3 = func_add_3.with_attr("Composite", "add3")
        func_add_3 = func_add_3.with_attr("PartitionedFromPattern", "add_add_")

        # Call add4 and add3
        pa = relay.var("pa", shape=(1, 16, 56, 56))
        pb = relay.var("pb", shape=(1, 16, 56, 56))
        pc = relay.var("pc", shape=(1, 16, 56, 56))
        pd = relay.var("pd", shape=(1, 16, 56, 56))
        pe = relay.var("pe", shape=(1, 16, 56, 56))
        pf = relay.var("pf", shape=(1, 16, 56, 56))

        call_func_add_3 = relay.Call(func_add_3, [pa, pb, pc])
        call_func_add_4 = relay.Call(func_add_4, [call_func_add_3, pd, pe, pf])

        return relay.Function([pa, pb, pc, pd, pe, pf], call_func_add_4)

    graph = before()
    result = run_opt_pass(graph,
                          relay.transform.MergeComposite(PATTERN_TABLE),
                          import_prelude=False)
    expected_graph = expect()
    expected_graph = run_opt_pass(expected_graph, relay.transform.InferType())
    assert tvm.ir.structural_equal(
        result, expected_graph, map_free_vars=True
    ), "Graph mismatch: output vs. expected\n{0}\n=====\n{1}".format(
        str(result), str(expected_graph))
    print("="*20)
    print(graph)
    print("="*20)
    print(result)
    print("="*20)
    print(expected_graph)


if __name__ == "__main__":
    test_reshape_transpose_reshape()
    test_addN()
