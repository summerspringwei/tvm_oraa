"""Define schedule strategy for topi operators"""

import tvm.relay.op as _op
from tvm.relay.op import strategy as _strategy
from tvm.target import override_native_generic_func

from ..topi import pixel_shuffle_cuda

# pylint: disable=unused-argument,
# pylint: disable=relative-beyond-top-level

def warp_pixel_shuffle(topi_compute):
    """warp pixel shuffle topi compute"""
    def _compute_pixel_shuffle(attrs, inputs, out_type):
        data = inputs[0]
        upscale_factor = attrs.upscale_factor
        args = [data, upscale_factor]
        return [topi_compute(*args)]
    return _compute_pixel_shuffle


@override_native_generic_func("pixel_shuffle_nchw_strategy")
def pixel_shuffle_nchw_strategy(attrs, inputs, output_type, target):
    """pixel shuffle cuda trategy"""
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        warp_pixel_shuffle(pixel_shuffle_cuda.pixel_shuffle_nchw),
        _strategy.wrap_topi_schedule(pixel_shuffle_cuda.schedule_pixel_shuffle_nchw),
        name="pixel_shuffle_cuda"
    )
    return strategy
