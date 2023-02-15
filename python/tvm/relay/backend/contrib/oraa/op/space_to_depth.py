"""Relay operators for space to depth for Open Research AI Architecture"""
from typing import Tuple

import tvm  # type: ignore
from tvm.relay.op import _make  # type: ignore


def oraa_space_to_depth(ifm: tvm.relay.Expr,
                       downscale_factor: Tuple[int, int] = (2, 2),
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
    downscale_factor : tuple of int, optional
        factor to increase spatial resolution by.
    ifm_layout : str, optional
        The layout of the Input Feature Map tensor. Can be "NCHW".
    ofm_layout : str, optional
        The layout of the Output Feature Map tensor. Can be "NCHW".

    Returns
    -------
    tvm.relay.Call
        A call to the oraa_space_to_depth op.

    """

    return _make.oraa_space_to_depth(ifm, downscale_factor, ifm_layout,
                                    ofm_layout)
