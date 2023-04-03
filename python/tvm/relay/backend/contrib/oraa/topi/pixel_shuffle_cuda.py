"""pixel shuffle compute for Open Research AI Architecture"""
# pylint: disable=invalid-name
from tvm import te, tir, autotvm
from tvm.topi import utils


def pixel_shuffle_nchw(input_tensor, upscale_factor):
    """ "Compute declaration for PixelShuffle with input's layout being NCHW"""
    if isinstance(upscale_factor, (list, tuple)):
        assert len(upscale_factor) == 2
        assert upscale_factor[0] == upscale_factor[1]
        upscale_factor = upscale_factor[0]
    assert isinstance(upscale_factor, int)

    batch, in_channel, in_height, in_width = utils.get_const_tuple(input_tensor.shape)
    out_channel = tir.indexdiv(tir.indexdiv(in_channel, upscale_factor), upscale_factor)
    out_height = in_height * upscale_factor
    out_width = in_width * upscale_factor
    output_tensor = te.compute(
        (batch, out_channel, out_height, out_width),
        lambda n, c, h, w: input_tensor[
            n,
            c * upscale_factor * upscale_factor
            + tir.indexmod(h, upscale_factor) * upscale_factor
            + tir.indexmod(w, upscale_factor),
            tir.indexdiv(h, upscale_factor),
            tir.indexdiv(w, upscale_factor),
        ],
        name="PixelShuffle",
    )
    return output_tensor


def pixel_unshuffle_nchw(input_tensor, downscale_factor):
    """ "Compute declaration for PixelUnshuffle with input's layout being NCHW"""
    if isinstance(downscale_factor, (list, tuple)):
        assert len(downscale_factor) == 2
        assert downscale_factor[0] == downscale_factor[1]
        downscale_factor = downscale_factor[0]
    assert isinstance(downscale_factor, int)

    batch, in_channel, in_height, in_width = utils.get_const_tuple(input_tensor.shape)
    out_channel = in_channel * downscale_factor * downscale_factor
    out_height = tir.indexdiv(in_height, downscale_factor)
    out_width = tir.indexdiv(in_width, downscale_factor)
    output_tensor = te.compute(
        (batch, out_channel, out_height, out_width),
        lambda n, c, h, w: input_tensor[
            n,
            tir.truncmod(c, in_channel),
            tir.truncdiv(tir.truncdiv(c, in_channel), downscale_factor) + (h * downscale_factor),
            tir.truncmod(tir.truncdiv(c, in_channel), downscale_factor) + (w * downscale_factor),
        ],
        name="PixelUnshuffle",
    )
    return output_tensor


@autotvm.register_topi_schedule("pixel_shuffle_nchw.cuda")
def schedule_pixel_shuffle_nchw(cfg, s, output):
    """pixel shuffle schedule for CUDA with NCHW layout"""
    n, c, h, w = s[output].op.axis

    cfg.define_know(
        "tile_c",
        [
            8,
        ],
    )
    cfg.define_know("tile_hw", [7, 7 * 7, 32, 64, 128, 256])
    cfg.define_knob("auto_unroll_max_step", [0, 512, 1500])

    nc = s[output].fuse(n, c)
    hw = s[output].fuse(h, w)
    nco, _ = s[output].split(nc, factor=cfg["tile_c"].val)
    hwo, hwi = s[output].split(hw, factor=cfg["tile_hw"].val)
    s[output].bind(nco, te.thread_axis("blockIdx.x"))
    s[output].bind(hwo, te.thread_axis("threadIdx.x"))
    s[output].pragma(hwi, "auto_unroll_max_step", cfg["auto_unroll_max_step"].val)

    return s
