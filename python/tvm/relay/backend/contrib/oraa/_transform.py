"""Open Research AI Architecture related feature registration"""
# pylint: disable=invalid-name,unused-argument, len-as-condition, too-many-nested-blocks,
# pylint: disable=too-many-arguments, no-else-return

from __future__ import absolute_import

import logging

from tvm.relay.op import op as _reg
from tvm.relay.backend.contrib import oraa
from .strategy import generic


@_reg.register_compute("contrib.oraa.pixel_shuffle")
def compute_pixel_shuffle(attrs, inputs, output_type):
    """Compute definition of pixel shuffle with NCHW layout"""
    logging.info("Call pixel shuffle compute")
    if attrs.ifm_layout.lower() == 'nchw':
        return [
            oraa.topi.pixel_shuffle_cuda.pixel_shuffle_nchw(
                inputs[0], attrs.upscale_factor)
        ]
    else:
        raise NotImplementedError("Only support NCHW layout on ORAA")


_reg.register_strategy("contrib.oraa.pixel_shuffle",
                       generic.pixel_shuffle_nchw_strategy)
