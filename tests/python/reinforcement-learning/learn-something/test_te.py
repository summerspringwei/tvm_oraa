import tvm

from tvm import te, tir


def expand_func(w_bit: tir.Var, off_tir:tir.Var, w_bit_tir: tir.Var,\
                 weight: te.Tensor, BITMASK: te.Tensor):
    shape1 = (32, )
    def smaller(off):
        const_32 = tvm.tir.const(32,"int32")
        return tvm.tir.LE(off_tir + w_bit_tir, const_32)
    def expand(i):
        off = tir.indexmod(i * w_bit, 32)
        small = smaller(off)
        return te.if_then_else(small, 
                                (
                                    tir.shift_right(weight[tir.indexdiv(tir.multiply(i, w_bit), 32)], off) &
                                    BITMASK[tir.min(32 - off, w_bit)]
                                ),
                                (
                                    tir.shift_right(weight[tir.indexdiv(tir.multiply(i, w_bit), 32)], off) &
                                    BITMASK[tir.min(32 - off, w_bit)]
                                )
                )
    
    return te.compute(shape=shape1, fcompute=expand, name="expand")


# 1. Pass tir.Var rather than IntImm
# 2. Bind Variables to tvm's function's args
# 3. Use tir.indexmod or tir.indexdiv rather than '/' or '%'
# 4. Use tir.if_then_else rather than python's if else

def test_expand_func():
    # w_bit = te.placeholder((1,), dtype="int32", name='w_bit')
    w_bit = tir.Var("w_bit", dtype="int32")
    off_tir = tvm.tir.Var("off","int32")
    w_bit_tir = tvm.tir.Var("w bit","int32")
    weight = te.placeholder((1024,), dtype='int32', name='weight')
    BITMASK = te.placeholder((1024,), dtype='int32', name='bitmask')
    output = expand_func(w_bit, off_tir, w_bit_tir, weight, BITMASK)
    s = te.create_schedule(output.op)
    print(tvm.lower(s, [w_bit, off_tir, w_bit_tir, weight, BITMASK], simple_mode=True))
    # func = tvm.build(s, [w_bit, weight, BITMASK], target='llvm', name="out", )
    # print(func)


if __name__=="__main__":
    test_expand_func()