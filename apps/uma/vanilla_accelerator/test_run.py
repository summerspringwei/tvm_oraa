import tvm
from tvm import relay
from backend import VanillaAcceleratorBackend
from tvm.relay import transform
import numpy as np


def create_conv2d_func(ishape=(1, 32, 14, 14), wshape=(32, 32, 3, 3), kernel_size=(3, 3), dtype="float32"):
    data0 = relay.var("data", shape=ishape, dtype=dtype)
    weight0 = relay.var("weight", shape=wshape, dtype=dtype)
    # hardware fixed param: strides=(1,1), padding=same, groups=1
    padding = (kernel_size[0]//2, kernel_size[1]//2)
    out = relay.nn.conv2d(data0, weight0, kernel_size=kernel_size, strides=(
        1, 1), groups=1, padding=padding)
    main_f = relay.Function([data0, weight0], out)
    return main_f


def main():
    ishape = (1, 1, 5, 10)  # NCHW
    wshape = (3, 1, 5, 5)  # OIHW
    kernel_size = (5, 5)
    f = create_conv2d_func(ishape=ishape, wshape=wshape,
                           kernel_size=kernel_size)
    print(f)
    mod = tvm.IRModule()
    mod["main"] = f
    mod = transform.InferType()(mod)

    uma_backend = VanillaAcceleratorBackend()
    uma_backend.register()
    mod = uma_backend.partition(mod)
    print(mod)
    target = tvm.target.Target(
        "vanilla_accelerator", host=tvm.target.Target("c"))
    real_inputs = np.ones(ishape).astype("float32")
    real_weights = np.ones(wshape).astype("float32")
    func = relay.create_executor(
        kind="aot", mod=mod, target=target
    ).evaluate()
    out = func(real_inputs, real_weights)
    print(out)


if __name__ == "__main__":
    main()
