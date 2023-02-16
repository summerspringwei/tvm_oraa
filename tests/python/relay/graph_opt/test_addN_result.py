"""
Test graph optimization rules for Open-Research-AI-Compiler
Add3/Add4 result tests
"""
import tvm
from tvm import relay
from tvm import te
from tvm import tir
from tvm.relay import transform
from tvm.relay.dataflow_pattern import is_op, wildcard
from tvm.relay.testing import run_opt_pass
from tvm.relay.backend.contrib.oraa import legalize
import numpy as np

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
    ("add4", make_add4_pattern()),
    ("add3", make_add3_pattern()),
]

# single add3 test
def test_add3():
    def add3_graph(ishape=(1,3,4,6)):
        x0 = relay.var("x0", shape=ishape, dtype="int8")
        x1 = relay.var("x1", shape=ishape, dtype="int8")
        x2 = relay.var("x2", shape=ishape, dtype="int8")
        r = x0 + x1 + x2
        return relay.Function([x0, x1, x2,], r)

    def concat_conv1x1_graph(ishape=(1, 3, 4, 6)):
        x0 = relay.var("x0", shape=ishape, dtype="int8")
        x1 = relay.var("x1", shape=ishape, dtype="int8")
        x2 = relay.var("x2", shape=ishape, dtype="int8")
        concat = relay.concatenate([x0, x1, x2], axis=1)
        in_channel = ishape[1]*3
        out_channel = ishape[1]
        # weight  OIHW
        weight_unit_np = np.identity(out_channel)[:,:,None,None]
        weight_np = np.concatenate((weight_unit_np,weight_unit_np,weight_unit_np),axis=1)
        weight = relay.const(weight_np.astype("int8"))
        conv = relay.nn.conv2d(concat, weight, kernel_size=(1, 1))
        return relay.Function([x0, x1, x2], conv)

    print("add3 test:")
    ishape = (3,8,24,24)  # NCHW
    in0 = np.random.randint(0,16,ishape).astype("int8")
    in1 = np.random.randint(0,16,ishape).astype("int8")
    in2 = np.random.randint(0,16,ishape).astype("int8")

    f = add3_graph(ishape)
    print("*"*20)
    print(f)
    mod = tvm.IRModule()
    mod["main"] = f
    mod = transform.InferType()(mod)
    target = tvm.target.Target("llvm")
    func = relay.create_executor(
        kind="graph", mod=mod, target=target
    ).evaluate()
    out = func(in0,in1,in2)
    f2 = concat_conv1x1_graph(ishape=ishape)
    print("*"*20)
    print(f2)
    mod2 = tvm.IRModule()
    mod2["main"] = f2
    mod2 = transform.InferType()(mod2)
    target = tvm.target.Target("llvm")
    func2 = relay.create_executor(
        kind="graph", mod=mod2, target=target
    ).evaluate()
    out2 = func2(in0,in1,in2)
    assert np.allclose(out.asnumpy(), out2.asnumpy())
    # add3 legalized concat+conv1x1
    annotated = run_opt_pass(f,
                          relay.transform.MergeComposite(PATTERN_TABLE),
                          import_prelude=False)
    rewrited = legalize.transform_oraa_function(annotated)
    rewrited = run_opt_pass(rewrited, relay.transform.InferType())
    print("*"*20)
    print(rewrited)
    mod3 = tvm.IRModule()
    mod3["main"] = rewrited
    mod3 = transform.InferType()(mod3)
    target = tvm.target.Target("llvm")
    func3 = relay.create_executor(
        kind="graph", mod=mod3, target=target
    ).evaluate()
    out3 = func3(in0,in1,in2)
    assert np.allclose(out.asnumpy(), out3.asnumpy())

# single add4 test
def test_add4():
    def add4_graph(ishape=(1,3,4,6)):
        x0 = relay.var("x0", shape=ishape, dtype="int8")
        x1 = relay.var("x1", shape=ishape, dtype="int8")
        x2 = relay.var("x2", shape=ishape, dtype="int8")
        x3 = relay.var("x3", shape=ishape, dtype="int8")
        r = x0 + x1 + x2 + x3
        return relay.Function([x0, x1, x2, x3], r)

    def concat_conv1x1_graph(ishape=(1, 3, 4, 6)):
        x0 = relay.var("x0", shape=ishape, dtype="int8")
        x1 = relay.var("x1", shape=ishape, dtype="int8")
        x2 = relay.var("x2", shape=ishape, dtype="int8")
        x3 = relay.var("x3", shape=ishape, dtype="int8")
        concat = relay.concatenate([x0, x1, x2, x3], axis=1)
        in_channel = ishape[1]*4
        out_channel = ishape[1]
        # weight  OIHW
        weight_unit_np = np.identity(out_channel)[:,:,None,None]
        weight_np = np.concatenate((weight_unit_np,weight_unit_np,weight_unit_np,weight_unit_np),axis=1)
        weight = relay.const(weight_np.astype("int8"))
        conv = relay.nn.conv2d(concat, weight, kernel_size=(1, 1))
        return relay.Function([x0, x1, x2, x3], conv)

    print("add4 test:")
    ishape = (3,16,12,24)  # NCHW
    in0 = np.random.randint(0,16,ishape).astype("int8")
    in1 = np.random.randint(0,16,ishape).astype("int8")
    in2 = np.random.randint(0,16,ishape).astype("int8")
    in3 = np.random.randint(0,16,ishape).astype("int8")
    # naive add4
    f = add4_graph(ishape)
    print("*"*20)
    print(f)
    mod = tvm.IRModule()
    mod["main"] = f
    mod = transform.InferType()(mod)
    target = tvm.target.Target("llvm")
    func = relay.create_executor(
        kind="graph", mod=mod, target=target
    ).evaluate()
    out = func(in0,in1,in2,in3)
    # straight concat+conv1x1
    f2 = concat_conv1x1_graph(ishape=ishape)
    print("*"*20)
    print(f2)
    mod2 = tvm.IRModule()
    mod2["main"] = f2
    mod2 = transform.InferType()(mod2)
    target = tvm.target.Target("llvm")
    func2 = relay.create_executor(
        kind="graph", mod=mod2, target=target
    ).evaluate()
    out2 = func2(in0,in1,in2,in3)
    assert np.allclose(out.asnumpy(), out2.asnumpy())
    # add4 legalized concat+conv1x1
    annotated = run_opt_pass(f,
                          relay.transform.MergeComposite(PATTERN_TABLE),
                          import_prelude=False)
    rewrited = legalize.transform_oraa_function(annotated)
    rewrited = run_opt_pass(rewrited, relay.transform.InferType())
    print("*"*20)
    print(rewrited)
    mod3 = tvm.IRModule()
    mod3["main"] = rewrited
    mod3 = transform.InferType()(mod3)
    target = tvm.target.Target("llvm")
    func3 = relay.create_executor(
        kind="graph", mod=mod3, target=target
    ).evaluate()
    out3 = func3(in0,in1,in2,in3)
    assert np.allclose(out.asnumpy(), out3.asnumpy())


# a combination of add3 and add4 test
def test_add11():
    def add11_graph(ishape=(1,3,4,6)):
        x0 = relay.var("x0", shape=ishape, dtype="int8")
        x1 = relay.var("x1", shape=ishape, dtype="int8")
        x2 = relay.var("x2", shape=ishape, dtype="int8")
        x3 = relay.var("x3", shape=ishape, dtype="int8")
        x4 = relay.var("x4", shape=ishape, dtype="int8")
        x5 = relay.var("x5", shape=ishape, dtype="int8")
        x6 = relay.var("x6", shape=ishape, dtype="int8")
        x7 = relay.var("x7", shape=ishape, dtype="int8")
        x8 = relay.var("x8", shape=ishape, dtype="int8")
        x9 = relay.var("x9", shape=ishape, dtype="int8")
        x10 = relay.var("x10", shape=ishape, dtype="int8")
        r = x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9 + x10
        return relay.Function([x0, x1, x2, x3,x4,x5,x6,x7,x8,x9,x10], r)


    print("add11 test:")
    ishape = (3,16,12,24)  # NCHW
    in0 = np.random.randint(0,16,ishape).astype("int8")
    in1 = np.random.randint(0,16,ishape).astype("int8")
    in2 = np.random.randint(0,16,ishape).astype("int8")
    in3 = np.random.randint(0,16,ishape).astype("int8")
    in4 = np.random.randint(0,16,ishape).astype("int8")
    in5 = np.random.randint(0,16,ishape).astype("int8")
    in6 = np.random.randint(0,16,ishape).astype("int8")
    in7 = np.random.randint(0,16,ishape).astype("int8")
    in8 = np.random.randint(0,16,ishape).astype("int8")
    in9 = np.random.randint(0,16,ishape).astype("int8")
    in10 = np.random.randint(0,16,ishape).astype("int8")
    # naive add4
    f = add11_graph(ishape)
    print("*"*10,"add11_graph","*"*10)
    print(f)
    mod = tvm.IRModule()
    mod["main"] = f
    mod = transform.InferType()(mod)
    target = tvm.target.Target("llvm")
    func = relay.create_executor(
        kind="graph", mod=mod, target=target
    ).evaluate()
    out = func(in0,in1,in2,in3,in4,in5,in6,in7,in8,in9,in10)
    # add11 -> legalized concat+conv1x1
    annotated = run_opt_pass(f,
                          relay.transform.MergeComposite(PATTERN_TABLE),
                          import_prelude=False)
    print("*"*10,"annotated graph","*"*10)
    print(annotated)
    rewrited = legalize.transform_oraa_function(annotated)
    rewrited = run_opt_pass(rewrited, relay.transform.InferType())
    print("*"*10,"legalized graph","*"*10)
    print(rewrited)
    mod3 = tvm.IRModule()
    mod3["main"] = rewrited
    mod3 = transform.InferType()(mod3)
    target = tvm.target.Target("llvm")
    func3 = relay.create_executor(
        kind="graph", mod=mod3, target=target
    ).evaluate()
    out3 = func3(in0,in1,in2,in3,in4,in5,in6,in7,in8,in9,in10)
    assert np.allclose(out.asnumpy(), out3.asnumpy())


if __name__ == "__main__":
    test_add3()
    test_add4()
    test_add11()
    print("success")