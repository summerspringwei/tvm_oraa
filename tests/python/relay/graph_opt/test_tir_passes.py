import tvm
from tvm import tir, te
from tvm.ir.module import IRModule
from tvm.script import tir as T
from tvm.driver.build_module import schedule_to_module
import numpy as np

# @tvm.script.ir_module
# class LoopPartitionModule:
#     @T.prim_func
#     def main(a: T.handle, b: T.handle):
#         # We exchange data between function by handles, which are similar to pointer.
#         T.func_attr({"global_symbol": "main", "tir.noalias": True})
#         # Create buffer from handles.
#         A = T.match_buffer(a, (64,), dtype="float32")
#         B = T.match_buffer(b, (64,), dtype="float32")
#         with T.block("root"):
#           for i in range(4):
#               for j in range(16):
#                 # A block is an abstraction for computation.
#                 with T.block("B"):
#                     # Define a spatial block iterator and bind it to value i.
#                     vi, vj = T.axis.remap("SS", [i, j])
#                     if tir.likely(vi * 4 + vj < 50):
#                       B[vi * 16 + vj] = A[vi * 16 + vj] + 1.0


# @T.prim_func
# def func_loop_partition(a: T.handle, b: T.handle):
#     # We exchange data between function by handles, which are similar to pointer.
#     T.func_attr({"global_symbol": "main", "tir.noalias": True})
#     # Create buffer from handles.
#     A = T.match_buffer(a, (64,), dtype="float32")
#     B = T.match_buffer(b, (64,), dtype="float32")
#     with T.block("root"):
#       for i in range(4):
#           for j in range(16):
#             # A block is an abstraction for computation.
#             with T.block("B"):
#                 # Define a spatial block iterator and bind it to value i.
#                 vi, vj = T.axis.remap("SS", [i, j])
#                 if tir.likely(vi * 4 + vj < 50):
#                   B[vi * 16 + vj] = A[vi * 16 + vj] + 1.0


@T.prim_func
def concat_func_3(
    placeholder: T.Buffer[(1, 64, 28, 28), "int8"],
    placeholder_1: T.Buffer[(1, 32, 28, 28), "int8"],
    placeholder_2: T.Buffer[(1, 32, 28, 28), "int8"],
    T_concat: T.Buffer[(1, 128, 28, 28), "int8"],
) -> None:
    placeholder_flat = T.buffer_decl([50176], "int8", data=placeholder.data)
    placeholder_1_flat = T.buffer_decl([25088], "int8", data=placeholder_1.data)
    placeholder_2_flat = T.buffer_decl([25088], "int8", data=placeholder_2.data)
    T_concat_flat = T.buffer_decl([100352], "int8", data=T_concat.data)
    for i1 in T.serial(128, annotations={"pragma_loop_partition_hint": 1}):
        for i2, i3 in T.grid(28, 28):
            if 96 <= i1:
                T_concat_flat[i1 * 784 + i2 * 28 + i3] = placeholder_2_flat[
                    i1 * 784 + i2 * 28 + i3 - 75264
                ]
            if 64 <= i1 and i1 < 96:
                T_concat_flat[i1 * 784 + i2 * 28 + i3] = placeholder_1_flat[
                    i1 * 784 + i2 * 28 + i3 - 50176
                ]
            if i1 < 64:
                T_concat_flat[i1 * 784 + i2 * 28 + i3] = placeholder_flat[i1 * 784 + i2 * 28 + i3]


def partition_from_scheduled_tir(prim_func, pass_cfg):
    with tvm.transform.PassContext(config=pass_cfg):
        mod = IRModule.from_expr(prim_func)
        mod = tvm.tir.transform.LowerOpaqueBlock()(mod)
        mod = tvm.tir.transform.FlattenBuffer()(mod)
        mod = tvm.tir.transform.LoopPartition()(mod)
        mod = tvm.tir.transform.Simplify()(mod)
        mod = tvm.tir.transform.RemoveNoOp()(mod)
        return mod



mod = partition_from_scheduled_tir(
        concat_func_3, {"tir.LoopPartition": {"partition_const_loop": True}}
)

print(mod.script())



@tvm.script.ir_module
class VectorizeModule:
    @T.prim_func
    def main(a: T.handle, b: T.handle):
        # We exchange data between function by handles, which are similar to pointer.
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # Create buffer from handles.
        A = T.match_buffer(a, (40,), dtype="float32")
        B = T.match_buffer(b, (40,), dtype="float32")
        with T.block("root"):
          for i in range(4):
              for j in range(16):
                # A block is an abstraction for computation.
                with T.block("B"):
                    # Define a spatial block iterator and bind it to value i.
                    vi, vj = T.axis.remap("SS", [i, j])
                    B[vi * 16 + vj] = A[vi * 16 + vj] + 1.0



def test_vectorize():
  ir_module = VectorizeModule
  print(type(ir_module))
  print(ir_module.script())
  mod = tvm.tir.transform.VectorizeLoop(True)(ir_module)
  print(mod.script())


def test_multi_loop():
    ib = tvm.tir.ir_builder.create()
    m = te.size_var("m")
    n = te.size_var("n")
    with ib.for_range(0, 4, "i") as i:
        with ib.for_range(0, n, "j") as j:
            with ib.for_range(0, m, "k") as k:
                with ib.if_scope(ib.likely(i * m + j + k < n)):
                    ib.emit(tvm.tir.Evaluate(m))
                with ib.else_scope():
                    ib.emit(tvm.tir.Evaluate(n))
    stmt = ib.get()

    mod = tvm.IRModule.from_expr(tvm.tir.PrimFunc([n, m], stmt))
    mod = tvm.tir.transform.LoopPartition()(mod)
    print(mod)
    stmt = tvm.tir.transform.Simplify()(mod)["main"].body



def test_vthread_vectorized():
    """Use of vthread is compatible with vector allocations"""

    @T.prim_func
    def before_func():
        vthread = T.env_thread("vthread")
        T.launch_thread(vthread, 4)
        B_data = T.allocate([4], "int32", "shared")
        B = T.buffer_decl([4], "int32", data=B_data, scope="shared")
        B[0:4] = T.broadcast(vthread, 4)

    @T.prim_func
    def expected_func():
        B_data = T.allocate([4], "int32x4", "shared")
        B = T.buffer_decl([4], "int32x4", data=B_data, scope="shared")
        B[T.Mul(0, 4) / 4] = T.broadcast(0, 4)
        B[T.Mul(1, 4) / 4] = T.broadcast(1, 4)
        B[T.Mul(2, 4) / 4] = T.broadcast(2, 4)
        B[T.Mul(3, 4) / 4] = T.broadcast(3, 4)

    before_mod = tvm.IRModule.from_expr(before_func)
    intermediate_mod = tvm.tir.transform.InjectVirtualThread()(before_mod)
    after_mod = tvm.tir.transform.StorageRewrite()(intermediate_mod)
    after_func = after_mod["main"]

    tvm.ir.assert_structural_equal(after_func, expected_func)



def test_storage_share():
    m = te.var("m")
    l = te.var("l")
    A = te.placeholder((m, l), name="A")
    num_stage = 5
    B = A
    for t in range(num_stage):
        B = te.compute((m, l), lambda i, j: B[i, j] + (t + 1), name="A%d" % t)

    s = te.create_schedule(B.op)
    mod = schedule_to_module(s, [A, B])
    mod = tvm.tir.transform.StorageFlatten(64)(mod)
    print(mod)
    mod = tvm.tir.transform.Simplify()(mod)
    mod = tvm.tir.transform.StorageRewrite()(mod)
    print(mod)
    stmt = mod["main"].body

    # verify only have one allocations.
    # verify inplace folding works
    num_alloc = [0]

    def verify(n):
        if isinstance(n, tvm.tir.Allocate):
            num_alloc[0] += 1

    tvm.tir.stmt_functor.post_order_visit(stmt, verify)
    assert num_alloc[0] == 1


def test_remove_no_op_with_invalid_extent():
    @T.prim_func
    def main(A: T.Buffer[(16), "int32"], B: T.Buffer[(16), "int32"]) -> None:
        for i in T.serial(16):
            for j in T.serial(i - 20):
                B[i] = A[i] + j

    mod = tvm.ir.module.IRModule.from_expr(main)
    print(mod.script())
    ret = tvm.tir.transform.RemoveNoOp()(mod)["main"]
    print(ret)
    # assert isinstance(ret, tvm.tir.Evaluate)



def test_rewrite_Select():
    ib = tvm.tir.ir_builder.create()
    A = ib.allocate("float32", 100, name="A", scope="global")
    i = te.var("i")
    y = tvm.tir.Select(i > 1, A[i - 1], 1.0)

    mod = tvm.IRModule.from_expr(tvm.tir.PrimFunc([i], tvm.tir.Evaluate(y)))
    yy = tvm.tir.transform.RewriteUnsafeSelect()(mod)["main"].body.value

    z = tvm.tir.Select(tvm.tir.Select(i > 1, A[i - 1], 1.0) > 0.0, A[i], 0.1)
    mod = tvm.IRModule.from_expr(tvm.tir.PrimFunc([i], tvm.tir.Evaluate(z)))
    zz = tvm.tir.transform.RewriteUnsafeSelect()(mod)["main"].body.value

    a = tvm.tir.Select(tvm.tir.floordiv(i, 4) > 10, y, z)

    mod = tvm.IRModule.from_expr(tvm.tir.PrimFunc([i], tvm.tir.Evaluate(a)))
    print(mod.script())
    mod = tvm.tir.transform.RewriteUnsafeSelect()(mod)
    aa = tvm.tir.transform.RewriteUnsafeSelect()(mod)["main"].body.value
    print(mod.script())
    builtin_if_then_else = tvm.ir.Op.get("tir.if_then_else")

    assert yy.op.same_as(builtin_if_then_else)
    assert yy.op.same_as(builtin_if_then_else)
    assert isinstance(aa, tvm.tir.Select)


# @tvm.testing.requires_llvm
def test_in_bounds_const_loop_partition_llvm():
    with tvm.transform.PassContext(
        config={
            "tir.instrument_bound_checkers": True,
            "tir.LoopPartition": {"partition_const_loop": True},
        }
    ):
        n = 21
        A = te.placeholder((n,), name="A")
        B = te.placeholder((n,), name="B")

        T = te.compute((n,), lambda i: A[i] + B[i])
        s = te.create_schedule(T.op)
        xo, xi = s[T].split(T.op.axis[0], factor=4)
        lowered_func = tvm.lower(s, [A, B, T], "llvm", simple_mode=False)
        dev = tvm.cpu(0)
        print(lowered_func.script())
        f = tvm.build(s, [A, B, T], "llvm")
        a = tvm.nd.array(np.random.uniform(size=(n,)).astype(A.dtype), dev)
        b = tvm.nd.array(np.random.uniform(size=(n,)).astype(B.dtype), dev)
        t = tvm.nd.empty((n,), T.dtype, dev)
        f(a, b, t)


if __name__=="__main__":
  # test_loop_partition()
  #  test_vectorize()
  # test_multi_loop()
  #   pass
#   test_storage_share()
#   test_remove_no_op_with_invalid_extent()
#   test_rewrite_Select()
  test_in_bounds_const_loop_partition_llvm()
