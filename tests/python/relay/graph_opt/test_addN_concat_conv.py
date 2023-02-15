"""
Testing replace AddN with Concat+Conv1x1 for Open-Research-AI-Compiler
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
        # print(weight_unit_np)
        weight_np = np.concatenate((weight_unit_np,weight_unit_np,weight_unit_np),axis=1)
        # print(weight_np)
        weight = relay.const(weight_np.astype("int8"))
        conv = relay.nn.conv2d(concat, weight, kernel_size=(1, 1))
        return relay.Function([x0, x1, x2], conv)

    ishape = (3,8,24,24)  # NCHW
    in0 = np.random.randint(0,16,ishape).astype("int8")
    in1 = np.random.randint(0,16,ishape).astype("int8")
    in2 = np.random.randint(0,16,ishape).astype("int8")

    f = add3_graph(ishape)
    mod = tvm.IRModule()
    mod["main"] = f
    mod = transform.InferType()(mod)
    target = tvm.target.Target("llvm")
    func = relay.create_executor(
        kind="graph", mod=mod, target=target
    ).evaluate()
    out = func(in0,in1,in2)

    f2 = concat_conv1x1_graph(ishape=ishape)
    mod2 = tvm.IRModule()
    mod2["main"] = f2
    mod2 = transform.InferType()(mod2)
    target = tvm.target.Target("llvm")
    func2 = relay.create_executor(
        kind="graph", mod=mod2, target=target
    ).evaluate()
    out2 = func2(in0,in1,in2)
    assert np.allclose(out.asnumpy(), out2.asnumpy())

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

    ishape = (3,16,12,24)  # NCHW
    in0 = np.random.randint(0,16,ishape).astype("int8")
    in1 = np.random.randint(0,16,ishape).astype("int8")
    in2 = np.random.randint(0,16,ishape).astype("int8")
    in3 = np.random.randint(0,16,ishape).astype("int8")

    f = add4_graph(ishape)
    mod = tvm.IRModule()
    mod["main"] = f
    mod = transform.InferType()(mod)
    target = tvm.target.Target("llvm")
    func = relay.create_executor(
        kind="graph", mod=mod, target=target
    ).evaluate()
    out = func(in0,in1,in2,in3)

    f2 = concat_conv1x1_graph(ishape=ishape)
    mod2 = tvm.IRModule()
    mod2["main"] = f2
    mod2 = transform.InferType()(mod2)
    target = tvm.target.Target("llvm")
    func2 = relay.create_executor(
        kind="graph", mod=mod2, target=target
    ).evaluate()
    out2 = func2(in0,in1,in2,in3)
    assert np.allclose(out.asnumpy(), out2.asnumpy())



if __name__ == "__main__":
    test_add3()
    test_add4()
    print("success")