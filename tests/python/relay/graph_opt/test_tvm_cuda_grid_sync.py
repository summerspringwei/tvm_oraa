import tvm
from tvm.ir.module import IRModule
from tvm.script import tir as T
import numpy as np


@tvm.script.ir_module
class MyModule:
    @T.prim_func
    def main(a: T.handle, b: T.handle):
        # We exchange data between function by handles, which are similar to pointer.
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # Create buffer from handles.
        A = T.match_buffer(a, (32,32), dtype="float32")
        B = T.match_buffer(b, (32,32), dtype="float32")
        with T.block("root"):
          for i in range(32):
              for j in range(32):
                # A block is an abstraction for computation.
                with T.block("B"):
                    # Define a spatial block iterator and bind it to value i.
                    vi, vj = T.axis.remap("SS", [i, j])
                    B[vi, vj] = A[vi, vj] + 1.0
              T.evaluate(T.call_intrin("handle", "tir.grid_sync"))
              # T.evaluate(T.call_intrin("handle", "tir.tvm_storage_sync", "global", "threadIdx.x==0", "32" ))


def test_tvm_cuda_grid_sync():
  ir_module = MyModule
  print(type(ir_module))
  print(ir_module.script())

  sch = tvm.tir.Schedule(ir_module)
  print(type(sch))

  # Get block by its name
  block_b = sch.get_block("B")
  # Get loops surrounding the block
  (i, j) = sch.get_loops(block_b)
  sch.bind(i, "blockIdx.x")
  sch.bind(j, "threadIdx.x")
  print(sch.mod.script())

  ctx = tvm.cuda(0)
  cuda_mod = tvm.build(sch.mod, target="cuda")
  print(cuda_mod.imported_modules[0].get_source())
  # cuda_a = tvm.nd.array(np.zeros((32, 32)).astype("float32"), ctx)
  # cuda_b = tvm.nd.array(np.zeros((32, 32)).astype("float32"), ctx)
  # cuda_mod(cuda_a, cuda_b)
  # print(cuda_a)
  # print(cuda_b)


if __name__=="__main__":
   test_tvm_cuda_grid_sync()