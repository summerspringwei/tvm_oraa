import tvm
from tvm import tir
from tvm.ir.module import IRModule
import tvm.testing
from tvm import te
from tvm.script import tir as T
from tvm.topi import utils
import numpy as np

def add4_nchw(in0,in1,in2,in3):
    batch,channel,height,width = utils.get_const_tuple(in0.shape)
    output_tensor = te.compute(
        (batch, channel, height, width),
        lambda n, c, h, w: in0[n,c,h,w]+in1[n,c,h,w]+in2[n,c,h,w]+in3[n,c,h,w],
        name="Add4")
    return output_tensor
    
def test_unroll_coreid(input_shape=[4,16,4,4]):
    @T.prim_func
    def origin2(in0: T.Buffer[(4,16,4,4), "int8"], in1: T.Buffer[(4,16,4,4), "int8"], in2: T.Buffer[(4,16,4,4), "int8"], in3: T.Buffer[(4,16,4,4), "int8"],
               Add4: T.Buffer[(4,16,4,4), "int8"]):
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        for n in T.thread_binding(4, thread="cthread"):
            for c,h,w in T.grid(16,4,4):
                with T.block("Add4"):
                    T.reads(in0[n, c, h, w], in1[n, c, h, w], in2[n, c, h, w], in3[n, c, h, w])
                    T.writes(Add4[n, c, h, w])
                    Add4[n, c, h, w] = in0[n, c, h, w] + in1[n, c, h, w] + in2[n, c, h, w] + in3[n, c, h, w]
    origin_from_te = tvm.IRModule.from_expr(origin2)
    sch = tir.Schedule(origin_from_te)
    (n, c, h, w) = sch.get_loops("Add4")
    sch.reorder(c,h,w,n)
    sch.bind(n, "cthread")
    sched_mod = sch.mod
    print(sched_mod.script())
    seq = tvm.transform.Sequential(
            [
                tvm.tir.transform.UnifyThreadBinding(),
                tvm.tir.transform.UnrollCthread(),
            ]
        )
    aftermod = seq(sched_mod)
    print(aftermod.script())


@T.prim_func
def prim_func_add4(in0: T.Buffer[(4,16,4,4), "int8"], in1: T.Buffer[(4,16,4,4), "int8"], in2: T.Buffer[(4,16,4,4), "int8"], in3: T.Buffer[(4,16,4,4), "int8"],
               Add4: T.Buffer[(4,16,4,4), "int8"]):
    T.func_attr({"global_symbol": "main", "tir.noalias": True})
    for n, c,h,w in T.grid(4,16,4,4):
        with T.block("Add4"):
        # v_n, v_c, v_h, v_w = T.axis.remap("SSSS", [n, c, h, w])
            T.reads(in0[n, c, h, w], in1[n, c, h, w], in2[n, c, h, w], in3[n, c, h, w])
            T.writes(Add4[n, c, h, w])
            Add4[n, c, h, w] = in0[n, c, h, w] + in1[n, c, h, w] + in2[n, c, h, w] + in3[n, c, h, w]
            
@T.prim_func
def prim_func_remap_add4(in0: T.Buffer[(4,16,4,4), "int8"], in1: T.Buffer[(4,16,4,4), "int8"], in2: T.Buffer[(4,16,4,4), "int8"], in3: T.Buffer[(4,16,4,4), "int8"],
               Add4: T.Buffer[(4,16,4,4), "int8"]):
    T.func_attr({"global_symbol": "main", "tir.noalias": True})
    for n, c,h,w in T.grid(4,16,4,4):
        with T.block("Add4"):
            v_n, v_c, v_h, v_w = T.axis.remap("SSSS", [n, c, h, w])
            T.reads(in0[v_n, v_c, v_h, v_w], in1[v_n, v_c, v_h, v_w], in2[v_n, v_c, v_h, v_w], in3[v_n, v_c, v_h, v_w])
            T.writes(Add4[v_n, v_c, v_h, v_w])
            Add4[v_n, v_c, v_h, v_w] = in0[v_n, v_c, v_h, v_w] + in1[v_n, v_c, v_h, v_w] + in2[v_n, v_c, v_h, v_w] + in3[v_n, v_c, v_h, v_w]

def test_unroll_coreid2():
    ir_module = tvm.IRModule.from_expr(prim_func_remap_add4)
    # print(ir_module)
    sch = tir.Schedule(ir_module)
    (n, c, h, w) = sch.get_loops("Add4")
    sch.reorder(c,h,w,n)
    # sch.bind(n, "blockIdx.x")
    sch.bind(c, "threadIdx.x")
    sched_mod = sch.mod
    print(sched_mod.script())
    seq = tvm.transform.Sequential(
            [
                # tvm.tir.transform.UnifyThreadBinding(),
                tvm.tir.transform.UnrollCthread(),
            ]
        )
    aftermod = seq(sched_mod)
    print(aftermod.script())
             
def test_unroll_multi_thread():
    ir_module = tvm.IRModule.from_expr(prim_func_add4)
    # print(ir_module)
    sch = tir.Schedule(ir_module)
    (n, c, h, w) = sch.get_loops("Add4")
    sch.reorder(c,h,w,n)
    sch.bind(n, "blockIdx.x")
    sch.bind(c, "threadIdx.x")
    sched_mod = sch.mod
    print(sched_mod.script())
    seq = tvm.transform.Sequential(
            [
                # tvm.tir.transform.UnifyThreadBinding(),
                tvm.tir.transform.UnrollCthread(),
            ]
        )
    aftermod = seq(sched_mod)
    print(aftermod.script())
    




if __name__ == "__main__":
    print("unroll1:")
    test_unroll_coreid()
    print("unroll2:")
    test_unroll_coreid2()
    print("unroll3:")
    test_unroll_multi_thread()
