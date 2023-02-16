"""
Test graph optimization rules for Open-Research-AI-Compiler
Add3 graph substitute tests
"""

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




PATTERN_TABLE = [
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
    import numpy as np
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

    def expect2(ishape=(1, 16, 56, 56),dtype="int8"):
        a = relay.var("a", shape=ishape, dtype=dtype)
        b = relay.var("b", shape=ishape, dtype=dtype)
        c = relay.var("c", shape=ishape, dtype=dtype)
        concat = relay.concatenate([a,b,c], axis=1)
        in_channel = ishape[1]*3
        out_channel = ishape[1]
        # weight  OIHW
        weight_unit_np = np.identity(out_channel)[:,:,None,None]
        # print(weight_unit_np)
        weight_np = np.concatenate((weight_unit_np,weight_unit_np,weight_unit_np),axis=1)
        # print(weight_np)
        weight = relay.const(weight_np.astype(dtype))
        conv = relay.nn.conv2d(concat, weight, kernel_size=(1, 1))
        return relay.Function([a,b,c], conv)

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
    rewrited = run_opt_pass(rewrited, relay.transform.InferType())
    expected2_graph = expect2()
    expected2_graph = run_opt_pass(expected2_graph, relay.transform.InferType())
    assert tvm.ir.structural_equal(
        rewrited, expected2_graph, map_free_vars=True
    ), "Graph mismatch: output vs. expected\n{0}\n=====\n{1}".format(
        str(rewrited), str(expected2_graph))
    print(rewrited)


if __name__ == "__main__":
    test_add3()
    print("success")
