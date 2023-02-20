import tvm
import numpy as np
from tvm import relay,te,testing
from tvm.topi.utils import *

def matmul_te(A,B):
    dtype="float32"
    target="llvm"
    dev = tvm.cpu(0)

    # Algorithm
    k = te.reduce_axis((0, K), "k")
    C = te.compute((M, N), lambda m, n: te.sum(A[m, k] * B[k, n], axis=k), name="C")
    return C

vM = te.size_var("M")
vK = te.size_var("K")
vN = te.size_var("N")
pA = te.placeholder((vM, vK), name="A")
pB = te.placeholder((vK, vN), name="B")

M = 256
K = 256
N = 256
dtype="float32"

# Random generated tensor for testing
a = tvm.nd.array(np.ones(shape=(M, K)).astype(dtype))
b = tvm.nd.array(np.ones(shape=(K, N)).astype(dtype))
c = tvm.nd.array(np.zeros((M, N), dtype=dtype))
answer = np.matmul(a.numpy(), b.numpy())
pC = matmul_te(A=pA, B=pB)
# Default schedule
s: te.schedule.Schedule = te.create_schedule(pC.op)

def cpu_gemm(pA,pB,pC):

    # print("-----tir code-----")
    # print(tvm.lower(s, [pA, pB, pC], simple_mode=True))

    cpu_func = tvm.build(s, [pA, pB, pC], target="llvm", name="matmul")
    cpu_func(a,b,c)
    tvm.testing.assert_allclose(c.numpy(), answer, rtol=1e-5)
    evaluator = cpu_func.time_evaluator(cpu_func.entry_name,dev=tvm.cpu(0), number=1)
    print("Baseline CPU: %f" % evaluator(a, b, c).mean)
    # print("-----CPU code-----")
    # print(cpu_func.get_source())

cpu_gemm(pA,pB,pC)


def gpu_gemm(pA,pB,pC):
    # C[m,n] += A[m,k]*B[k,n]

    block_size = 16
    tx, ty, tk = 16, 16, 16
    ctx = tvm.cuda()
    def split(stage, axis, factors):
        axes=[]
        for f in reversed(factors):
            axis, x = stage.split(axis, f)
            axes.append(x)
        return list(reversed(axes+[axis]))

    def bind_thread(stage, axes, tags):
        for axis, tag in zip(axes, tags):
            stage.bind(axis, te.thread_axis(tag))

    def optimize_read_cache(shared, local):
        s[shared].compute_at(s[C_local], ko)
        s[local].compute_at(s[C_local], ki)
        y,x = s[shared].op.axis
        yo, yi = s[shared].split(y, nparts=block_size)
        xo, xi = s[shared].split(x, nparts=block_size)
        s[shared].reorder(yo, xo, yi, xi)
        bind_thread(s[shared], (yo,xo), ("threadIdx.y", "threadIdx.x"))

    A_shared = s.cache_read(pA, "shared", [pC])
    A_local  = s.cache_read(A_shared, "local", [pC])
    B_shared = s.cache_read(pB, "shared", [pC])
    B_local  = s.cache_read(B_shared, "local", [pC])
    C_local = s.cache_write(pC, "local")
    # Split each axis into block axis, thread axis, and inner axis
    x, y = s[pC].op.axis
    xb,xo,xi = split(s[pC],x,(block_size, tx))
    yb,yo,yi = split(s[pC],y,(block_size, ty))
    s[pC].reorder(xb, yb, xo, yo, xi, yi)
    # we bind yb to blockIdx.x instead of blockIdx.y
    bind_thread(s[pC], [yb, xb, yo, xo],("blockIdx.x", "blockIdx.y", "threadIdx.x", "threadIdx.y"))
    # schedule C_local
    s[C_local].compute_at(s[pC], yo)
    yi,xi = s[C_local].op.axis
    k, = s[C_local].op.reduce_axis
    ko,ki = s[C_local].split(k, tk)
    s[C_local].reorder(ko, ki, yi, xi)

    optimize_read_cache(A_shared, A_local)
    optimize_read_cache(B_shared, B_local)
    print("-"*10,"tir code","-"*10)
    print(tvm.lower(s, [pA, pB, pC], simple_mode=True))



    cuda_func = tvm.build(s, [pA, pB, pC], target="cuda", name="matmul")
    dev_module = cuda_func.imported_modules[0]
    print("-"*10,"GPU code","-"*10)
    print(dev_module.get_source())
    dev = tvm.cuda(0)
    a2 = tvm.nd.array(np.ones(shape=(M, K)).astype(dtype),device=dev)
    b2 = tvm.nd.array(np.ones(shape=(K, N)).astype(dtype),device=dev)
    c2 = tvm.nd.array(np.zeros((M, N), dtype=dtype),device=dev)
    cuda_func(a2,b2,c2)
    # print(c2)
    tvm.testing.assert_allclose(c2.numpy(), answer, rtol=1e-5)
    evaluator = cuda_func.time_evaluator(cuda_func.entry_name,dev=tvm.cuda(0), number=1)
    print("GPU shared: %f" % evaluator(a2, b2, c2).mean)

gpu_gemm(pA,pB,pC)
