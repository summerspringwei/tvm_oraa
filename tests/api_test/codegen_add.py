from dla2.client import get_remote_runtime
from dla2.core.flags import TensorFormat
import numpy as np
api = get_remote_runtime()
buf = api.malloc(1536)
max_core = 8
def main_kernel0(in0, in1, T_add):
    in0_shared = api.declare_tensor(buf, 0, [1,8,8,8], TensorFormat.NCHW)
    in1_shared = api.declare_tensor(buf, 512, [1,8,8,8], TensorFormat.NCHW)
    out_shared = api.declare_tensor(buf, 1024, [1,8,8,8], TensorFormat.NCHW)
    for ax1_0 in range(2):
        for ax2_0 in range(2):
            for ax3_0 in range(2):
                in0_sliced = in0[0:1,(ax1_0*8):(ax1_0*8+8),(ax2_0*8):(ax2_0*8+8),(ax3_0*8):(ax3_0*8+8)]
                api.write_to_tensor(in0_shared, in0_sliced)
                in1_sliced = in1[0:1,(ax1_0*8):(ax1_0*8+8),(ax2_0*8):(ax2_0*8+8),(ax3_0*8):(ax3_0*8+8)]
                api.write_to_tensor(in1_shared, in1_sliced)
                api.add(tlc=0,in0=in0_shared,in1=in1_shared,out=out_shared)
                api.synchronize(dry_run=False)
                T_add[0:1,(ax1_0*8):(ax1_0*8+8),(ax2_0*8):(ax2_0*8+8),(ax3_0*8):(ax3_0*8+8)] = api.read_from_tensor(out_shared)

A_data = np.random.randint(0,128,[1,16,16,16],dtype="int8")
B_data = np.random.randint(0,128,[1,16,16,16],dtype="int8")
Out_data = np.zeros([1,16,16,16],dtype="int8")
main_kernel0(A_data,B_data,Out_data)




