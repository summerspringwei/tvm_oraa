"""
Testing for the graph optimization (Add3/Add4) for Open-Research-AI-Compiler
Replace AddN into combinations of Add3/Add4, which map to concat+conv1x1
"""
import tvm
from tvm import relay
from tvm import te
from tvm import tir
from tvm.relay.dataflow_pattern import is_op, wildcard
from tvm.relay.testing import run_opt_pass
from tvm.relay.backend.contrib.oraa import legalize


def make_add_relu_pattern(have_relu=True):
    r"""Create a pattern to match the following graph.

     add
      |
    relu(optional)
    """
    add_node = wildcard() + wildcard()
    if (have_relu):
        add_node = is_op("nn.relu")(add_node)
    return add_node


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


PATTERN_TABLE = [
    #NOTE: order is important, first come, first match
    ("add4", make_add4_pattern()),
    ("add3", make_add3_pattern()),
]


def test_add3():
    r"""Test composite function is correctly produced with 3-element add.

        a  b
         \/
         add c                  a b c
           \/                   \   / 
           add        ====>      add3

    """
    def before():
        a = relay.var("a", shape=(1, 16, 56, 56), dtype="int8")
        b = relay.var("b", shape=(1, 16, 56, 56), dtype="int8")
        c = relay.var("c", shape=(1, 16, 56, 56), dtype="int8")
        r = a + b + c
        return relay.Function([a, b, c], r)

    def expect():
        # Declare add3 function
        a = relay.var("a", shape=(1, 16, 56, 56), dtype="int8")
        b = relay.var("b", shape=(1, 16, 56, 56), dtype="int8")
        c = relay.var("c", shape=(1, 16, 56, 56), dtype="int8")
        add3 = a + b + c
        func_add_3 = relay.Function([a, b, c], add3)
        func_add_3 = func_add_3.with_attr("Composite", "add3")
        func_add_3 = func_add_3.with_attr("PartitionedFromPattern", "add_add_")

        pa = relay.var("pa", shape=(1, 16, 56, 56), dtype="int8")
        pb = relay.var("pb", shape=(1, 16, 56, 56), dtype="int8")
        pc = relay.var("pc", shape=(1, 16, 56, 56), dtype="int8")
        call_func_add_3 = relay.Call(func_add_3, [pa, pb, pc])
        return relay.Function([pa, pb, pc], call_func_add_3)

    print("=" * 20)
    print("add3 test:")
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
    print(graph)
    print("=" * 20)
    # rewrite
    rewrited = legalize.transform_oraa_function(result)
    print(rewrited)


def test_add4():
    r"""Test composite function is correctly produced with 3-element add.

        a  b
         \/
         add c                  
           \/                   
           add d                a b c d
            \ /                 \     / 
            add        ====>      add4
    """
    def before():
        a = relay.var("a", shape=(1, 16, 56, 56), dtype="int8")
        b = relay.var("b", shape=(1, 16, 56, 56), dtype="int8")
        c = relay.var("c", shape=(1, 16, 56, 56), dtype="int8")
        d = relay.var("d", shape=(1, 16, 56, 56), dtype="int8")
        r = a + b + c + d
        return relay.Function([a, b, c, d], r)

    def expect():
        # Declare add3 function
        a = relay.var("a", shape=(1, 16, 56, 56), dtype="int8")
        b = relay.var("b", shape=(1, 16, 56, 56), dtype="int8")
        c = relay.var("c", shape=(1, 16, 56, 56), dtype="int8")
        d = relay.var("d", shape=(1, 16, 56, 56), dtype="int8")
        add4 = a + b + c + d
        func_add_4 = relay.Function([a, b, c, d], add4)
        func_add_4 = func_add_4.with_attr("Composite", "add4")
        func_add_4 = func_add_4.with_attr("PartitionedFromPattern",
                                          "add_add_add_")

        pa = relay.var("pa", shape=(1, 16, 56, 56), dtype="int8")
        pb = relay.var("pb", shape=(1, 16, 56, 56), dtype="int8")
        pc = relay.var("pc", shape=(1, 16, 56, 56), dtype="int8")
        pd = relay.var("pd", shape=(1, 16, 56, 56), dtype="int8")
        call_func_add_4 = relay.Call(func_add_4, [pa, pb, pc, pd])
        return relay.Function([pa, pb, pc, pd], call_func_add_4)

    print("=" * 20)
    print("add4 test:")
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
    print(graph)
    print("=" * 20)
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
        a = relay.var("a", shape=(1, 16, 56, 56), dtype="int8")
        b = relay.var("b", shape=(1, 16, 56, 56), dtype="int8")
        c = relay.var("c", shape=(1, 16, 56, 56), dtype="int8")
        d = relay.var("d", shape=(1, 16, 56, 56), dtype="int8")
        e = relay.var("e", shape=(1, 16, 56, 56), dtype="int8")
        f = relay.var("f", shape=(1, 16, 56, 56), dtype="int8")

        r = a + b + c + d + e + f

        return relay.Function([a, b, c, d, e, f], r)

    def expect():
        # Declare add4 function
        a = relay.var("a", shape=(1, 16, 56, 56), dtype="int8")
        b = relay.var("b", shape=(1, 16, 56, 56), dtype="int8")
        c = relay.var("c", shape=(1, 16, 56, 56), dtype="int8")
        d = relay.var("d", shape=(1, 16, 56, 56), dtype="int8")
        add_4 = a + b + c + d
        func_add_4 = relay.Function([a, b, c, d], add_4)
        func_add_4 = func_add_4.with_attr("Composite", "add4")
        func_add_4 = func_add_4.with_attr("PartitionedFromPattern",
                                          "add_add_add_")

        # Declare add3 function
        e = relay.var("e", shape=(1, 16, 56, 56), dtype="int8")
        f = relay.var("f", shape=(1, 16, 56, 56), dtype="int8")
        g = relay.var("g", shape=(1, 16, 56, 56), dtype="int8")
        add_3 = e + f + g
        func_add_3 = relay.Function([e, f, g], add_3)
        func_add_3 = func_add_3.with_attr("Composite", "add3")
        func_add_3 = func_add_3.with_attr("PartitionedFromPattern", "add_add_")

        # Call add4 and add3
        pa = relay.var("pa", shape=(1, 16, 56, 56), dtype="int8")
        pb = relay.var("pb", shape=(1, 16, 56, 56), dtype="int8")
        pc = relay.var("pc", shape=(1, 16, 56, 56), dtype="int8")
        pd = relay.var("pd", shape=(1, 16, 56, 56), dtype="int8")
        pe = relay.var("pe", shape=(1, 16, 56, 56), dtype="int8")
        pf = relay.var("pf", shape=(1, 16, 56, 56), dtype="int8")

        call_func_add_3 = relay.Call(func_add_3, [pa, pb, pc])
        call_func_add_4 = relay.Call(func_add_4, [call_func_add_3, pd, pe, pf])

        return relay.Function([pa, pb, pc, pd, pe, pf], call_func_add_4)

    print("=" * 20)
    print("add6 test:")
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
    print(graph)
    print("=" * 20)
    # rewrite
    rewrited = legalize.transform_oraa_function(result)
    print(rewrited)


def test_addN_big():
    def demo_graph():
        # generate x0 to x16
        x0 = relay.var("x0", shape=(1, 16, 56, 56), dtype="int8")
        x1 = relay.var("x1", shape=(1, 16, 56, 56), dtype="int8")
        x2 = relay.var("x2", shape=(1, 16, 56, 56), dtype="int8")
        x3 = relay.var("x3", shape=(1, 16, 56, 56), dtype="int8")
        x4 = relay.var("x4", shape=(1, 16, 56, 56), dtype="int8")
        x5 = relay.var("x5", shape=(1, 16, 56, 56), dtype="int8")
        x6 = relay.var("x6", shape=(1, 16, 56, 56), dtype="int8")
        x7 = relay.var("x7", shape=(1, 16, 56, 56), dtype="int8")
        x8 = relay.var("x8", shape=(1, 16, 56, 56), dtype="int8")
        x9 = relay.var("x9", shape=(1, 16, 56, 56), dtype="int8")
        x10 = relay.var("x10", shape=(1, 16, 56, 56), dtype="int8")
        x11 = relay.var("x11", shape=(1, 16, 56, 56), dtype="int8")
        x12 = relay.var("x12", shape=(1, 16, 56, 56), dtype="int8")
        x13 = relay.var("x13", shape=(1, 16, 56, 56), dtype="int8")
        x14 = relay.var("x14", shape=(1, 16, 56, 56), dtype="int8")
        x15 = relay.var("x15", shape=(1, 16, 56, 56), dtype="int8")
        x16 = relay.var("x16", shape=(1, 16, 56, 56), dtype="int8")

        r = x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + \
            x9 + x10 + x11 + x12 + x13 + x14 + x15 + x16

        return relay.Function([
            x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14,
            x15, x16
        ], r)

    print("=" * 20)
    print("add17 test:")
    graph = demo_graph()
    result = run_opt_pass(graph,
                          relay.transform.MergeComposite(PATTERN_TABLE),
                          import_prelude=False)
    result = run_opt_pass(result, relay.transform.InferType())
    print(graph)
    print("=" * 20)
    print(result)
    print("=" * 20)
    # rewrite
    rewrited = legalize.transform_oraa_function(result)
    print(rewrited)
    seq = tvm.transform.Sequential(
        [
            relay.transform.RemoveUnusedFunctions(),
            relay.transform.FoldConstant(),
        ]
    )
    mod = tvm.IRModule.from_expr(rewrited)
    mod = seq(mod)
    print("=" * 20)
    print(mod)

    


if __name__ == "__main__":
    # test_add2()
    test_add3()
    test_add4()
    test_addN()
    test_addN_big()
    print("success")
