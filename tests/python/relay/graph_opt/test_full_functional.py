import numpy as np

import tvm
from tvm import relay, te, tir
from tvm import IRModule
from tvm import meta_schedule as ms
from tvm.target import Target
from tvm.topi.utils import get_const_tuple
from tvm.relay.backend.contrib.oraa import op
from tvm.relay.backend.contrib.oraa.topi import pixel_shuffle_cuda
from tvm.relay.backend.contrib.oraa.tir.tensor_intrin import oraa_cuda
from tvm.script import tir as T
import numpy as np

def test_tensorize_oraa_add(input_shape):
    a = te.placeholder(input_shape, dtype="int8", name="add_a")
    b = te.placeholder(input_shape, dtype="int8", name="add_b")
    out = te.compute(
        input_shape,
        lambda vn, vc, vh, vw: a[vn, vc, vh, vw] + b[vn, vc, vh, vw],
    )
    func = te.create_prim_func([a, b, out])
    mod = IRModule({"main": func})
    sch = tir.Schedule(mod)
    # print(sch.mod)
    block = sch.get_block("compute")
    (n, c, h, w) = sch.get_loops(block)
    no, ni = sch.split(n, factors=[None, 2])
    co, ci = sch.split(c, factors=[None, 8])
    ho, hi = sch.split(h, factors=[None, 4])
    wo, wi = sch.split(w, factors=[None, 4])
    sch.reorder(no, co, ho, wo, ni, ci, hi, wi)
    block_inner = sch.blockize(ni)
    a_shared = sch.cache_read(block_inner, 0, "shared")
    b_shared = sch.cache_read(block_inner, 1, "shared")
    e_shared = sch.cache_write(block_inner, 0, "shared")
    sch.compute_at(a_shared, wo)
    sch.compute_at(b_shared, wo)
    sch.reverse_compute_at(e_shared, wo)
    ani, _, _, _ = sch.get_loops(a_shared)[-4:]
    sch.tensorize(ani, oraa_cuda.ORAA_LDG2S_N2C8H4W4_INT8_INTRIN)
    bni, _, _, _ = sch.get_loops(b_shared)[-4:]
    sch.tensorize(bni, oraa_cuda.ORAA_LDG2S_N2C8H4W4_INT8_INTRIN)
    e_n, _, _, _ = sch.get_loops(e_shared)[-4:]
    sch.tensorize(e_n, oraa_cuda.ORAA_STS2G_N2C8H4W4_INT8_INTRIN)
    sch.tensorize(ni, oraa_cuda.ORAA_Add2_N2C8H4W4_INTRIN)
    sch.bind(no, "blockIdx.x")
    target = Target("oraa")
    # target = 'cuda'
    func = tvm.build(sch.mod, [a, b], target=target)
    print(func)
    print(func.imported_modules[0].get_source())
    a_data = tvm.nd.array(np.ones(input_shape,dtype="int8"),device=tvm.oraa())
    b_data = tvm.nd.array(np.ones(input_shape,dtype="int8"),device=tvm.oraa())
    c_data = tvm.nd.array(np.zeros(input_shape,dtype="int8"),device=tvm.oraa())
    func(a_data,b_data,c_data)


if __name__ == "__main__":
    test_tensorize_oraa_add((2, 16, 8, 8))
