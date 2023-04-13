import numpy as np

import tvm
from tvm import relay, te, topi, tir
from tvm import IRModule
from tvm import meta_schedule as ms
from tvm.target import Target
from tvm.topi.utils import get_const_tuple
from tvm.relay.backend.contrib.oraa import op
from tvm.relay.backend.contrib.oraa.topi import pixel_shuffle_cuda
from tvm.relay.backend.contrib.oraa.tir.tensor_intrin import oraa_cuda
from tvm.script import tir as T

@T.prim_func
def rowsum_cross_thread_reduction(a: T.handle, b: T.handle) -> None:
    T.func_attr({"global_symbol": "main", "tir.noalias": True})
    A = T.match_buffer(a, (128, 128))
    B = T.match_buffer(b, (128,))
    for i0 in T.serial(0, 128):
        for i1 in T.thread_binding(0, 128, thread="threadIdx.x"):
            with T.block("B"):
                vi, vk = T.axis.remap("SR", [i0, i1])
                with T.init():
                    B[vi] = 0.0
                B[vi] = B[vi] + A[vi, vk]
                
def test_func():
    A = te.placeholder((128,128),name="A")
    B = te.placeholder((128,),name="B")
    mod = tvm.IRModule.from_expr(rowsum_cross_thread_reduction)
    sch = tir.Schedule(mod)
    print(sch.mod)
    _, k = sch.get_loops(sch.get_block("B"))
    sch.bind(k, "threadIdx.x")
    print(sch.mod)
    target = Target("cuda")
    func = tvm.build(sch.mod, [A, B], target=target)
    print(func)
    print(func.imported_modules[0].get_source())
    
test_func()