"""Relay operators for add3/4 for Open Research AI Architecture"""
from typing import Tuple

import tvm  # type: ignore
from tvm.relay.op import _make  # type: ignore


def oraa_add3(in0: tvm.relay.Expr,
              in1: tvm.relay.Expr,
              in2: tvm.relay.Expr,
              ifm_layout: str = "NCHW",
              ofm_layout: str = "NCHW") -> tvm.relay.Call:
    """
    This Relay operator corresponds to the hardware-implemented add3 operator
    found on Open Research AI Architecture. It accepts NCHW
    format for the input data (Input Feature Map, or IFM).

    Parameters
    ----------
    in0, in1, in2 : tvm.relay.Expr
        The Input Feature Map tensor (IFM).
    ifm_layout : str, optional
        The layout of the Input Feature Map tensor. Can be "NCHW".
    ofm_layout : str, optional
        The layout of the Output Feature Map tensor. Can be "NCHW".

    Returns
    -------
    tvm.relay.Call
        A call to the ethosu_conv2d op.

    """
    return _make.oraa_add3(in0, in1, in2, ifm_layout,
                                    ofm_layout)


def oraa_add4(in0: tvm.relay.Expr,
              in1: tvm.relay.Expr,
              in2: tvm.relay.Expr,
              in3: tvm.relay.Expr,
              ifm_layout: str = "NCHW",
              ofm_layout: str = "NCHW") -> tvm.relay.Call:
    """
    This Relay operator corresponds to the hardware-implemented add4 operator
    found on Open Research AI Architecture. It accepts NCHW
    format for the input data (Input Feature Map, or IFM).

    Parameters
    ----------
    in0, in1, in2, in3 : tvm.relay.Expr
        The Input Feature Map tensor (IFM).
    ifm_layout : str, optional
        The layout of the Input Feature Map tensor. Can be "NCHW".
    ofm_layout : str, optional
        The layout of the Output Feature Map tensor. Can be "NCHW".

    Returns
    -------
    tvm.relay.Call
        A call to the ethosu_conv2d op.

    """
    return _make.oraa_add4(in0, in1, in2, in3, ifm_layout,
                                    ofm_layout)
