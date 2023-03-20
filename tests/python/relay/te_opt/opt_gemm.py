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



def cpu_gemm(pA,pB,pC,s:te.schedule.Schedule):

    # print("-----tir code-----")
    # print(tvm.lower(s, [pA, pB, pC], simple_mode=True))

    cpu_func = tvm.build(s, [pA, pB, pC], target="llvm", name="matmul")
    cpu_func(a,b,c)
    tvm.testing.assert_allclose(c.numpy(), answer, rtol=1e-5)
    evaluator = cpu_func.time_evaluator(cpu_func.entry_name,dev=tvm.cpu(0), number=1)
    print("Baseline CPU: %f" % evaluator(a, b, c).mean)
    # print("-----CPU code-----")
    # print(cpu_func.get_source())




def gpu_gemm_naive(pA,pB,pC,s:te.schedule.Schedule):
    # C[m,n] += A[m,k]*B[k,n]
    ctx = tvm.cuda()
    block_size = 4

    # schedule
    y, x = s[pC].op.axis
    k = s[pC].op.reduce_axis[0]
    
    yo, yi = s[pC].split(y, factor=block_size)
    xo, xi = s[pC].split(x, factor=block_size)

    s[pC].bind(yo, te.thread_axis("blockIdx.y"))
    s[pC].bind(xo, te.thread_axis("blockIdx.x"))
    s[pC].bind(yi, te.thread_axis("threadIdx.y"))
    s[pC].bind(xi, te.thread_axis("threadIdx.x"))
    # s[pC].reorder(yo, xo, k, yi, xi)

    # print("-"*10,"tir code","-"*10)
    # print(tvm.lower(s, [pA, pB, pC], simple_mode=True))
    cuda_func = tvm.build(s, [pA, pB, pC], target="cuda", name="matmul")
    dev_module = cuda_func.imported_modules[0]
    # print("-"*10,"GPU code","-"*10)
    # print(dev_module.get_source())
    dev = tvm.cuda(0)
    a2 = tvm.nd.array(np.ones(shape=(M, K)).astype(dtype),device=dev)
    b2 = tvm.nd.array(np.ones(shape=(K, N)).astype(dtype),device=dev)
    c2 = tvm.nd.array(np.zeros((M, N), dtype=dtype),device=dev)
    cuda_func(a2,b2,c2)
    # print(c2)
    tvm.testing.assert_allclose(c2.numpy(), answer, rtol=1e-5)
    evaluator = cuda_func.time_evaluator(cuda_func.entry_name,dev=tvm.cuda(0), number=1)
    print("GPU naive: %f" % evaluator(a2, b2, c2).mean)


    
def gpu_gemm(A,B,C,s:te.schedule.Schedule):
    # C[m,n] += A[m,k]*B[k,n]
    C_i, C_j, C_k = tuple(C.op.axis) + tuple(C.op.reduce_axis)
    C_local, = s.cache_write([C], "local")
    C_local_i_c, C_local_j_c, C_local_k = tuple(C_local.op.axis) + tuple(C_local.op.reduce_axis)
    C_local_i_c_o_i, C_local_i_c_i = s[C_local].split(C_local_i_c, factor=4)
    C_local_i_c_o_o_i, C_local_i_c_o_i = s[C_local].split(C_local_i_c_o_i, factor=4)
    C_local_i_c_o_o_o_i, C_local_i_c_o_o_i = s[C_local].split(C_local_i_c_o_o_i, factor=1)
    C_local_i_c_o_o_o_o, C_local_i_c_o_o_o_i = s[C_local].split(C_local_i_c_o_o_o_i, factor=1)
    C_local_j_c_o_i, C_local_j_c_i = s[C_local].split(C_local_j_c, factor=1)
    C_local_j_c_o_o_i, C_local_j_c_o_i = s[C_local].split(C_local_j_c_o_i, factor=1)
    C_local_j_c_o_o_o_i, C_local_j_c_o_o_i = s[C_local].split(C_local_j_c_o_o_i, factor=64)
    C_local_j_c_o_o_o_o, C_local_j_c_o_o_o_i = s[C_local].split(C_local_j_c_o_o_o_i, factor=4)
    C_local_k_o_i, C_local_k_i = s[C_local].split(C_local_k, factor=1)
    C_local_k_o_o, C_local_k_o_i = s[C_local].split(C_local_k_o_i, factor=4)
    s[C_local].reorder(C_local_i_c_o_o_o_o, C_local_j_c_o_o_o_o, C_local_i_c_o_o_o_i, C_local_j_c_o_o_o_i, C_local_i_c_o_o_i, C_local_j_c_o_o_i, C_local_k_o_o, C_local_k_o_i, C_local_i_c_o_i, C_local_j_c_o_i, C_local_k_i, C_local_i_c_i, C_local_j_c_i)
    C_i_o_i, C_i_i = s[C].split(C_i, factor=16)
    C_i_o_o_i, C_i_o_i = s[C].split(C_i_o_i, factor=1)
    C_i_o_o_o, C_i_o_o_i = s[C].split(C_i_o_o_i, factor=1)
    C_j_o_i, C_j_i = s[C].split(C_j, factor=1)
    C_j_o_o_i, C_j_o_i = s[C].split(C_j_o_i, factor=64)
    C_j_o_o_o, C_j_o_o_i = s[C].split(C_j_o_o_i, factor=4)
    s[C].reorder(C_i_o_o_o, C_j_o_o_o, C_i_o_o_i, C_j_o_o_i, C_i_o_i, C_j_o_i, C_i_i, C_j_i)
    s[C_local].compute_at(s[C], C_j_o_i)
    B_shared = s.cache_read(B, "shared", [C_local])
    B_shared_ax0, B_shared_ax1 = tuple(B_shared.op.axis)
    s[B_shared].compute_at(s[C_local], C_local_k_o_o)
    A_shared = s.cache_read(A, "shared", [C_local])
    A_shared_ax0, A_shared_ax1 = tuple(A_shared.op.axis)
    s[A_shared].compute_at(s[C_local], C_local_k_o_o)
    C_i_o_o_o_j_o_o_o_fused = s[C].fuse(C_i_o_o_o, C_j_o_o_o)
    s[C].bind(C_i_o_o_o_j_o_o_o_fused, te.thread_axis("blockIdx.x"))
    C_i_o_o_i_j_o_o_i_fused = s[C].fuse(C_i_o_o_i, C_j_o_o_i)
    s[C].bind(C_i_o_o_i_j_o_o_i_fused, te.thread_axis("vthread"))
    C_i_o_i_j_o_i_fused = s[C].fuse(C_i_o_i, C_j_o_i)
    s[C].bind(C_i_o_i_j_o_i_fused, te.thread_axis("threadIdx.x"))
    B_shared_ax0_ax1_fused = s[B_shared].fuse(B_shared_ax0, B_shared_ax1)
    B_shared_ax0_ax1_fused_o, B_shared_ax0_ax1_fused_i = s[B_shared].split(B_shared_ax0_ax1_fused, factor=4)
    s[B_shared].vectorize(B_shared_ax0_ax1_fused_i)
    B_shared_ax0_ax1_fused_o_o, B_shared_ax0_ax1_fused_o_i = s[B_shared].split(B_shared_ax0_ax1_fused_o, factor=64)
    s[B_shared].bind(B_shared_ax0_ax1_fused_o_i, te.thread_axis("threadIdx.x"))
    A_shared_ax0_ax1_fused = s[A_shared].fuse(A_shared_ax0, A_shared_ax1)
    A_shared_ax0_ax1_fused_o, A_shared_ax0_ax1_fused_i = s[A_shared].split(A_shared_ax0_ax1_fused, factor=1)
    s[A_shared].vectorize(A_shared_ax0_ax1_fused_i)
    A_shared_ax0_ax1_fused_o_o, A_shared_ax0_ax1_fused_o_i = s[A_shared].split(A_shared_ax0_ax1_fused_o, factor=64)
    s[A_shared].bind(A_shared_ax0_ax1_fused_o_i, te.thread_axis("threadIdx.x"))
    s[C_local].pragma(C_local_i_c_o_o_o_o, "auto_unroll_max_step", 1024)
    s[C_local].pragma(C_local_i_c_o_o_o_o, "unroll_explicit", True)

    cuda_func = tvm.build(s, [A, B, C], target="cuda", name="matmul")
    dev_module = cuda_func.imported_modules[0]
    # print("-"*10,"GPU code","-"*10)
    # print(dev_module.get_source())
    dev = tvm.cuda(0)
    a2 = tvm.nd.array(np.ones(shape=(M, K)).astype(dtype),device=dev)
    b2 = tvm.nd.array(np.ones(shape=(K, N)).astype(dtype),device=dev)
    c2 = tvm.nd.array(np.zeros((M, N), dtype=dtype),device=dev)
    cuda_func(a2,b2,c2)
    # print(c2)
    tvm.testing.assert_allclose(c2.numpy(), answer, rtol=1e-5)
    evaluator = cuda_func.time_evaluator(cuda_func.entry_name,dev=tvm.cuda(0), number=1)
    print("GPU shared: %f" % evaluator(a2, b2, c2).mean)



if __name__ == "__main__":
    vM = te.size_var("M")
    vK = te.size_var("K")
    vN = te.size_var("N")
    pA = te.placeholder((vM, vK), name="A")
    pB = te.placeholder((vK, vN), name="B")

    M = 4096
    K = 4096
    N = 2048
    dtype="float32"

    # Random generated tensor for testing
    a = tvm.nd.array(np.ones(shape=(M, K)).astype(dtype))
    b = tvm.nd.array(np.ones(shape=(K, N)).astype(dtype))
    c = tvm.nd.array(np.zeros((M, N), dtype=dtype))
    answer = np.matmul(a.numpy(), b.numpy())
    pC = matmul_te(A=pA, B=pB)
    # Default schedule
    # s: te.schedule.Schedule = te.create_schedule(pC.op)
    # cpu_gemm(pA,pB,pC,s)
    # GPU sharemem schedule
    # s: te.schedule.Schedule = te.create_schedule(pC.op)
    # gpu_gemm_shared(pA,pB,pC,s)

    # GPU schedule
    s: te.schedule.Schedule = te.create_schedule(pC.op)
    gpu_gemm_naive(pA,pB,pC,s)
    s: te.schedule.Schedule = te.create_schedule(pC.op)
    gpu_gemm(A=pA,B=pB,C=pC,s=s)