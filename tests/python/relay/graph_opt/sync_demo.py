

import oraa_tf as api


def case_1():
    """Conv+Add, no need sync"""
    a = api.talloc([1, 16, 56, 56], api.TensorFormat.NCHW, offset=0)
    b = api.talloc([32, 16, 3, 3], api.TensorFormat.NCHW, offset=1*16*56*56)
    c = api.talloc([1, 32, 56, 56], api.TensorFormat.NCHW)

    # Filter with layout [OC, IC, KH, KW]
    b_0 = b[0:16, :, 3, 3]
    b_1 = b[16:32, :, 3, 3]
    c_0 = c[:, 0:16, 56, 56]
    c_1 = c[:, 16:32, 56, 56]

    api.conv(0, c_0, a, b_0)
    api.conv(1, c_1, a, b_1)

    d = api.talloc([1, 64, 56, 56], api.TensorFormat.NCHW)
    d_0 = d[:, 0:16, 56, 56]
    d_1 = d[:, 16:32, 56, 56]

    # No need to sync here
    api.add(0, d_0, d_0, c_0)
    api.add(1, d_1, d_1, c_1)

    out = api.read_from_tensor(d)
    score = api.synchronize()
    api.free()


def case_2():
    """Conv+Conv, need sync"""
    a = api.talloc([1, 16, 56, 56], api.TensorFormat.NCHW)
    b = api.talloc([32, 16, 3, 3], api.TensorFormat.OCICHW)
    c = api.talloc([1, 32, 56, 56], api.TensorFormat.NCHW)

    # Filter with layout [OC, IC, KH, KW]
    b_0 = b[0:16, :, 3, 3]
    b_1 = b[16:32, :, 3, 3]

    c_0 = c[:, 0:16, 56, 56]
    c_1 = c[:, 16:32, 56, 56]

    api.conv(0, c_0, a, b_0)
    api.conv(1, c_1, a, b_1)

    d = api.talloc([128, 64, 56, 56], api.TensorFormat.OCICHW)
    e = api.talloc([1, 128, 56, 56], api.TensorFormat.NCHW)
    d_0 = d[0:16, :, 3, 3]
    d_1 = d[16:32, :, 3, 3]
    e_0 = e[:, 0:32, 56, 56]
    e_1 = e[:, 32:64, 56, 56]

    # TODO(Chunwei Xia) Need sync here. Two ways:
    # (1) sync between sliced tensors
    api.write_barrier(0, c_0)
    api.write_barrier(1, c_1)
    api.write_barrier(0, c_0)
    api.write_barrier(1, c_1)
    api.write_barrier(0, c_0)
    api.write_barrier(1, c_1)
    api.write_barrier(0, c_0)
    api.write_barrier(1, c_1)
    
    # (2) directly sync the original tensor
    api.write_barrier(0, c)
    api.write_barrier(1, c)

    api.conv(0, e_0, c[0:1, :, :, :], d_0)
    api.conv(1, e_1, c, d_1)

    
def case_3():
    """Pipeline Conv-core0-Add-core1, conv-core0-Add-core1"""
    a = api.talloc([1, 16, 56, 56], api.TensorFormat.NCHW)
    b = api.talloc([64, 16, 3, 3], api.TensorFormat.OCICHW)
    c = api.talloc([1, 64, 56, 56], api.TensorFormat.NCHW)
    d = api.talloc([1, 64, 56, 56], api.TensorFormat.NCHW)

    b_0 = b[0:32, :, 3, 3]
    b_1 = b[32:64, :, 3, 3]
    c_0 = c[:, 0:32, 56, 56]
    c_1 = c[:, 32:64, 56, 56]
    d_0 = d[:, 0:32, 56, 56]
    d_1 = d[:, 32:64, 56, 56]

    # Pipeline between core0 and core 1
    api.conv(0, c_0, a, b_0)
    api.write_barrier(1, c_0)
    api.add(1, d_0, d_0, c_0)
    api.conv(0, c_1, a, b_1)
    api.write_barrier(1, c_1)
    api.add(1, d_1, d_1, c_1)
    

def case_4():
    """Pipeline Conv-core0-Add-core1, conv-core0-Add-core1"""
    a = api.talloc([1, 16, 56, 56], api.TensorFormat.NCHW)
    b = api.talloc([64, 16, 3, 3], api.TensorFormat.OCICHW)
    c = api.talloc([1, 64, 56, 56], api.TensorFormat.NCHW)
    d = api.talloc([1, 64, 56, 56], api.TensorFormat.NCHW)
    e = api.talloc([1, 64, 56, 56], api.TensorFormat.NCHW)

    b_0 = b[0:32, :, 3, 3]
    b_1 = b[32:64, :, 3, 3]
    c_0 = c[:, 0:32, 56, 56]
    c_1 = c[:, 32:64, 56, 56]
    d_0 = d[:, 0:32, 56, 56]
    d_1 = d[:, 32:64, 56, 56]
    e_0 = e[:, 0:32, 56, 56]
    e_1 = e[:, 32:64, 56, 56]

    # Pipeline between core0 and core 1
    api.conv(0, c_0, a, b_0)
    #TODO(Chunwei Xia) Can the softmax be executed in parallel with core1's add?
    api.softmax(0, e_0, c_0) # read c_0 here
    api.write_barrier(1, c_0)
    api.add(1, d_0, d_0, c_0)
    api.conv(0, c_1, a, b_1)
    api.write_barrier(1, c_1)
    api.add(1, d_1, d_1, c_1)


def case_5():
    """Conv+Reshape+Permute"""
    a = api.talloc([1, 16, 56, 56], api.TensorFormat.NCHW)
    b = api.talloc([64, 16, 3, 3], api.TensorFormat.OCICHW)
    c = api.talloc([1, 64, 56, 56], api.TensorFormat.NCHW)
    c_copy = api.talloc([1, 64, 56, 56], api.TensorFormat.NCHW)
    d = api.talloc([1, 16, 112, 112], api.TensorFormat.NCHW)
    
    b_0 = b[0:32, :, 3, 3]
    b_1 = b[32:64, :, 3, 3]
    c_0 = c[:, 0:32, 56, 56]
    c_1 = c[:, 32:64, 56, 56]
  
    # Pipeline between core0 and core 1
    api.conv(0, c_0, a, b_0)
    api.conv(1, c_1, a, b_1)
    # TODO(Chunwei) Avoid extra concat, two ways
    # (1)
    api.write_barrier(0, c_1)
    api.concat(0, c_copy, [c_0, c_1], 1)
    api.pixel_shuffle(0, d, c_copy, 2)

    # (2)
    api.write_barrier(0, c_0)
    api.write_barrier(1, c_1)
    api.pixel_shuffle(0, d, c, 2)
