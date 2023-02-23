import os

import numpy as np
import tvm
from tvm import te, auto_scheduler, testing
from tvm.topi.utils import *

@auto_scheduler.register_workload  # Note the auto_scheduler decorator
def matmul_add(M, K, N, dtype):
    A = te.placeholder((M, K), name="A", dtype=dtype)
    B = te.placeholder((K, N), name="B", dtype=dtype)
    # C = te.placeholder((M, N), name="C", dtype=dtype)

    k = te.reduce_axis((0, K), name="k")
    C = te.compute(
        (M, N),
        lambda i, j: te.sum(A[i, k] * B[k, j], axis=k),
        name="C",
        # attrs={"layout_free_placeholders": [B]},  # enable automatic layout transform for tensor B
    )

    return [A, B, C]


M = 4096
K = 4096
N = 2048
dtype="float32"
target = tvm.target.Target("cuda")
task = auto_scheduler.SearchTask(
    func=matmul_add, args=(M, K, N, dtype), target=target
)
# Inspect the computational graph
print("Computational DAG:")
print(task.compute_dag)

log_file = "gemm_cuda.json"
measure_ctx = auto_scheduler.LocalRPCMeasureContext(min_repeat_ms=300)
tune_option = auto_scheduler.TuningOptions(
    num_measure_trials=32,  # change this to 1000 to achieve the best performance
    runner=measure_ctx.runner,
    measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
    verbose=2,
)

# Run auto-tuning (search)
task.tune(tune_option)
# Apply the best schedule
sch, args = task.apply_best(log_file)
# Kill the measurement process
del measure_ctx

print("Best schedule:")
print(task.print_best(log_file))
# print("Lowered TIR:")
# print(tvm.lower(sch, args, simple_mode=True))

cuda_func = tvm.build(sch, args, target="cuda", name="matmul")
dev_module = cuda_func.imported_modules[0]
# print("-"*10,"GPU code","-"*10)
# print(dev_module.get_source())
dev = tvm.cuda(0)
a = tvm.nd.array(np.ones(shape=(M, K)).astype(dtype))
b = tvm.nd.array(np.ones(shape=(K, N)).astype(dtype))
c = tvm.nd.array(np.zeros((M, N), dtype=dtype))
answer = np.matmul(a.numpy(), b.numpy())
a2 = tvm.nd.array(np.ones(shape=(M, K)).astype(dtype),device=dev)
b2 = tvm.nd.array(np.ones(shape=(K, N)).astype(dtype),device=dev)
c2 = tvm.nd.array(np.zeros(shape=(M, N),dtype=dtype), device=dev)
cuda_func(a2,b2,c2)
# print(c2)
tvm.testing.assert_allclose(c2.numpy(), answer, rtol=1e-5)
evaluator = cuda_func.time_evaluator(cuda_func.entry_name,dev=tvm.cuda(0), number=1)
print("GPU shared: %f" % evaluator(a2, b2, c2).mean)