
import pickle

import tvm
from tvm import auto_scheduler, te
from tvm.target import Target
from tvm import auto_scheduler as ms


@auto_scheduler.register_workload
def test_matmul(m, n, k):
    a = te.placeholder((m, k), dtype="float16")
    b = te.placeholder((n, k), dtype="float16")
    rk = te.reduce_axis((0, k), 'rk')
    c = te.compute((m, n), lambda i, j: te.sum(a[i,k] * b[j, k], axis=[rk]))
    return [a, b, c]


def save_and_load_v11():
    target = Target("nvidia/nvidia-a100")
    task = tvm.auto_scheduler.SearchTask(func=test_matmul, args=[1024, 1024, 1024], target=target)
    pickle.dump(task, open("my-a100-test_matmul", 'wb'))
    print(task.compute_dag.tensors)
    prim_func = te.create_prim_func(task.compute_dag.tensors)
    print(prim_func.script())
    
    print("load")
    task = pickle.load(open("my-a100-test_matmul", 'rb'))
    print(task.compute_dag)
