"""Relay operators for pixel shuffle for Open Research AI Architecture"""
from typing import Tuple

import tvm  # type: ignore
from tvm.relay.op import _make  # type: ignore


def oraa_pixel_shuffle(ifm: tvm.relay.Expr,
                       upscale_factor: Tuple[int, int] = (2, 2),
                       ifm_layout: str = "NCHW",
                       ofm_layout: str = "NCHW") -> tvm.relay.Call:
    """
    This Relay operator corresponds to the hardware-implemented pixel shuffle operator
    found on Open Research AI Architecture. It accepts NCHW
    format for the input data (Input Feature Map, or IFM).

    Reference: https://pytorch.org/docs/stable/generated/torch.nn.PixelShuffle.html

    Parameters
    ----------
    ifm : tvm.relay.Expr
        The Input Feature Map tensor (IFM).
    upscale_factor : tuple of int, optional
        factor to increase spatial resolution by.
    ifm_layout : str, optional
        The layout of the Input Feature Map tensor. Can be "NHWC".
    ofm_layout : str, optional
        The layout of the Output Feature Map tensor. Can be "NHWC".

    Returns
    -------
    tvm.relay.Call
        A call to the ethosu_conv2d op.

    """

    return _make.oraa_pixel_shuffle(ifm, upscale_factor, ifm_layout,
                                    ofm_layout)
