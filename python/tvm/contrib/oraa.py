"""Utility to execute source code in oraa runtime in the system"""
import os

import tvm._ffi
from tvm.target import Target

from . import utils
from .._ffi.base import py_str
@tvm._ffi.register_func
def tvm_callback_oraa_compile(code: str):
    """directly execute python code"""
    # code only contains device function
    exec(code)
    